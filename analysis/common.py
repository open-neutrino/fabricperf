"""Shared helpers for FabricPerf analysis modules."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable


DEFAULT_TRACE_ROOT = os.environ.get("NEUTRINO_TRACEDIR", "./trace")
FABRICPERF_MODE_ENV = "FABRICPERF_MODE"
FABRICPERF_THROUGHPUT_PARTITIONS_ENV = "FABRICPERF_THROUGHPUT_PARTITIONS"
LATENCY_MODE = "latency"
MEMORY_MODE = "memory"
THROUGHPUT_MODE = "throughput"
VALID_MODES = (LATENCY_MODE, MEMORY_MODE, THROUGHPUT_MODE)
THROUGHPUT_MODES = (THROUGHPUT_MODE,)
THROUGHPUT_WORKGROUPS = 4
THROUGHPUT_CELLS_PER_WORKGROUP = 1024
THROUGHPUT_HEADER_CELLS = 2
THROUGHPUT_SLOTS = THROUGHPUT_WORKGROUPS * THROUGHPUT_CELLS_PER_WORKGROUP
THROUGHPUT_CAPTURE_CAPACITY = THROUGHPUT_CELLS_PER_WORKGROUP - THROUGHPUT_HEADER_CELLS
THROUGHPUT_BIN_NS = 8000
VALID_THROUGHPUT_PARTITIONS = (4,)
DEFAULT_THROUGHPUT_PARTITIONS = 4

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent


def fail(message: str) -> int:
    """Print one analyzer-owned error and return a shell failure code.

    Motivation: post-run analyzers are called by `neutrino`, so failures must
    be machine-simple and human-readable. Example: fail("missing trace")
    emits "[error] missing trace" on stderr and returns 1.
    """
    print(f"[error] {message}", file=sys.stderr)
    return 1


def fabricperf_mode(environ: dict[str, str] | None = None) -> str:
    """Return the selected FabricPerf analyzer mode.

    Motivation: analysis is split between latency traces and memory CSV merge
    output. Example: an unset FABRICPERF_MODE reports latency tables, while
    FABRICPERF_MODE=memory writes fabricperf_cupti.csv.
    """
    source = os.environ if environ is None else environ
    mode = source.get(FABRICPERF_MODE_ENV, LATENCY_MODE).strip().lower()
    if mode not in VALID_MODES:
        joined = ", ".join(VALID_MODES)
        raise ValueError(f"{FABRICPERF_MODE_ENV} must be one of: {joined}")
    return mode


def throughput_variant(environ: dict[str, str] | None = None) -> str:
    """Return the normalized throughput variant for analyzer output.

    Motivation: the public CSV schema still has a `variant` column, but the
    active throughput implementation is the direct-GMEM path. Example:
    FABRICPERF_MODE=throughput reports variant "gmem".
    """
    source = os.environ if environ is None else environ
    mode = fabricperf_mode(source)
    if mode == THROUGHPUT_MODE:
        return "gmem"
    return ""


def throughput_partitions(environ: dict[str, str] | None = None) -> int:
    """Return the requested throughput workgroup partition count.

    Motivation: the analyzer must decode the same partition layout that the
    runtime published. Example: `FABRICPERF_THROUGHPUT_PARTITIONS=4` decodes
    the current fixed four-slice probe layout.
    """
    source = os.environ if environ is None else environ
    raw = source.get(FABRICPERF_THROUGHPUT_PARTITIONS_ENV, str(DEFAULT_THROUGHPUT_PARTITIONS)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{FABRICPERF_THROUGHPUT_PARTITIONS_ENV} must be 4") from exc
    if value not in VALID_THROUGHPUT_PARTITIONS:
        raise ValueError(f"{FABRICPERF_THROUGHPUT_PARTITIONS_ENV} must be 4")
    return value


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a FabricPerf CSV into field names and row dictionaries.

    Motivation: memory mode rewrites one final CSV after CUPTI PM Sampling and
    Neutrino probe output are available. Example: PM rows keep NVLink fields
    while probe rows supply DRAM/XBAR byte rates.
    """
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            return list(reader.fieldnames or []), rows
    except OSError as exc:
        raise ValueError(f"could not read FabricPerf CSV {path}: {exc}") from exc


def parse_raw_metrics(raw: str) -> tuple[list[str], dict[str, str]]:
    """Parse the semicolon-delimited raw_metrics field while preserving order.

    Motivation: merged CSVs should add probe-backed values without reordering
    CUPTI metadata. Example: `cupti_backend=pm_sampling` stays before appended
    `probe_read_bytes=...` fields.
    """
    order: list[str] = []
    values: dict[str, str] = {}
    for part in raw.split(";"):
        if part == "":
            continue
        key, separator, value = part.partition("=")
        if key not in values:
            order.append(key)
        values[key] = value if separator else ""
    return order, values


def format_raw_metrics(order: list[str], values: dict[str, str]) -> str:
    """Format an ordered raw_metrics mapping back into the CSV field string.

    Motivation: raw_metrics carries both metric values and profiler metadata.
    Example: probe byte totals are appended after PM Sampling keys so readers
    keep seeing the familiar CUPTI prefix.
    """
    for key in values:
        if key not in order:
            order.append(key)
    return ";".join(
        f"{key}={values[key]}" if values[key] != "" else key
        for key in order
    )


def is_diagnostic_raw_metric(metric: str) -> bool:
    """Return True for raw_metrics keys that describe profiler bookkeeping.

    Motivation: the long metrics CSV should be easy to pivot over hardware
    counters, while PM/probe diagnostics belong in a separate file. Example:
    `pm_samples_total` is diagnostic, but `nvltx__bytes.sum` is a metric.
    """
    # Keep backend/control fields out of the metric/value counter table.
    exact_diagnostics = {
        "cupti_backend",
        "event_duration_s",
        "pm_samples_total",
        "pm_samples_populated",
        "pm_samples_completed",
        "pm_overflow",
        "pm_interval",
        "fabricperf_probe_backend",
        "probe_merge_key",
        "probe_records",
    }
    return metric in exact_diagnostics


def write_long_metrics_csv(trace_root: Path, rows: list[dict[str, str]]) -> int:
    """Write metric/value and diagnose/value rows from the final CUPTI CSV.

    Motivation: packed `raw_metrics` strings are hard to scan or pivot. Example:
    `nvltx__bytes.sum=68784128` goes to `fabricperf_cupti_metrics.csv`, while
    `cupti_backend=pm_sampling` goes to `fabricperf_cupti_diagnose.csv`.
    """
    metrics_path = trace_root / "fabricperf_cupti_metrics.csv"
    diagnose_path = trace_root / "fabricperf_cupti_diagnose.csv"
    fieldnames = ["rank", "device", "launch_index", "kernel", "metric", "value"]
    try:
        with metrics_path.open("w", encoding="utf-8", newline="") as metrics_handle, \
                diagnose_path.open("w", encoding="utf-8", newline="") as diagnose_handle:
            metrics_writer = csv.DictWriter(metrics_handle, fieldnames=fieldnames)
            diagnose_writer = csv.DictWriter(diagnose_handle, fieldnames=fieldnames)
            metrics_writer.writeheader()
            diagnose_writer.writeheader()
            for row in rows:
                order, values = parse_raw_metrics(row.get("raw_metrics", ""))
                for metric in order:
                    writer = diagnose_writer if is_diagnostic_raw_metric(metric) else metrics_writer
                    writer.writerow({
                        "rank": row.get("rank", ""),
                        "device": row.get("device", ""),
                        "launch_index": row.get("launch_index", ""),
                        "kernel": row.get("kernel", ""),
                        "metric": metric,
                        "value": values.get(metric, ""),
                    })
    except OSError as exc:
        return fail(f"failed to write long-form FabricPerf CSVs under {trace_root}: {exc}")
    return 0


def float_field(raw: str | None) -> float | None:
    """Parse a CSV numeric field without treating blanks as zero.

    Motivation: blank duration or Bps fields mean no usable runtime duration was
    recorded. Example: PM Sampling may leave duration_s blank if event timing
    could not be collected.
    """
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def number_field(value: float | int) -> str:
    """Format numeric CSV fields consistently with native FabricPerf rows.

    Motivation: analyzer-merged probe values should sort and parse like CUPTI
    values. Example: 1024 becomes "1024" while 1.5e6 keeps significant digits.
    """
    if isinstance(value, int):
        return str(value)
    return f"{value:.17g}"


def bps_from_bytes(byte_count: int, duration_s: float | None) -> str:
    """Return a Bps field from bytes and duration, or blank when unavailable.

    Motivation: existing wide CSV columns use rates, while probe records store
    byte counts. Example: 1024 bytes over 0.001 seconds reports 1024000 Bps.
    """
    if duration_s is None or duration_s <= 0.0:
        return ""
    return number_field(float(byte_count) / duration_s)


def write_wide_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write FabricPerf's wide CSV schema to one path.

    Motivation: one-pass merge rewrites the public root CSV after reading
    per-rank PM rows. Example: trace/fabricperf_cupti.csv receives merged rows.
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_args() -> argparse.Namespace:
    """Parse analyzer arguments for CLI and standalone use.

    Motivation: neutrino --plugin fabricperf passes <tracedir>, while manual
    runs should still work. Example: analyze.py uses $NEUTRINO_TRACEDIR or
    ./trace if no trace root is provided.
    """
    parser = argparse.ArgumentParser(
        description="Read FabricPerf latency traces or merge memory-mode CUPTI/probe output.",
    )
    parser.add_argument(
        "trace_roots",
        nargs="*",
        default=[DEFAULT_TRACE_ROOT],
        help="Trace roots or per-rank trace directories. Defaults to $NEUTRINO_TRACEDIR or ./trace.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Also print a per-rank validation summary.",
    )
    return parser.parse_args()


def resolve_trace_root(raw: str) -> Path:
    """Resolve a trace root from absolute, workspace-relative, or repo-relative input.

    Motivation: packaged analyzers can be called from arbitrary directories.
    Example: both ./trace and neutrino/tmp_trace_sendrecv_d6_ptp_latency_smoke
    resolve when run from the workspace.
    """
    path = Path(raw).expanduser()
    if path.exists():
        return path.resolve()

    workspace_path = (WORKSPACE_ROOT / path).resolve()
    if workspace_path.exists():
        return workspace_path

    repo_path = (REPO_ROOT / path).resolve()
    if repo_path.exists():
        return repo_path

    raise FileNotFoundError(f"trace root not found: {raw}")


def iter_trace_dirs(root: Path) -> Iterable[tuple[str, Path]]:
    """Yield per-rank trace directories under a root.

    Motivation: Neutrino writes one directory per traced rank, but users may pass
    either the parent or a single rank directory. Example: trace/ yields all
    children containing read.py, while trace/May27_... yields only itself.
    """
    if (root / "read.py").is_file() and (root / "result").is_dir():
        yield root.parent.name, root
        return

    children = sorted(
        child for child in root.iterdir() if child.is_dir() and (child / "read.py").is_file()
    )
    if not children:
        raise FileNotFoundError(f"no per-rank trace directories under: {root}")

    for child in children:
        yield root.name, child


def load_reader(trace_dir: Path) -> ModuleType:
    """Load Neutrino's generated read.py module for one rank trace.

    Motivation: the result schema is generated from the selected probe, so
    parsing must use the trace-local reader. Example: a trace with sr_latency
    records exposes reader.parse(result_file).
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    module_name = f"_neutrino_trace_read_{abs(hash(trace_dir))}"
    spec = importlib.util.spec_from_file_location(module_name, trace_dir / "read.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load generated reader from {trace_dir / 'read.py'}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_rank(trace_dir: Path) -> int:
    """Infer MPI rank from the FabricPerf preparation line in event.log.

    Motivation: result files do not encode MPI rank directly. Example: both
    "[fabricperf] prepared rank=0" and "[plugin] prepared rank=0" are accepted.
    """
    event_log = trace_dir / "event.log"
    rank_re = re.compile(
        r"(?:prepared(?:\s+non-MPI)?\s+rank=(\d+)\b|ready\s+backend=[^\n]*\brank=(\d+)\b)",
        re.IGNORECASE,
    )
    with event_log.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = rank_re.search(line)
            if match is not None:
                # Accept both latency-style "prepared rank=N" and memory-style
                # "ready backend=... rank=N" event lines.
                return int(next(group for group in match.groups() if group is not None))
    raise ValueError(f"could not infer rank from {event_log}")


def format_float(value: float) -> str:
    """Format integral floats without a decimal point and others to two places."""
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}"


def count_range(values: Iterable[int]) -> str:
    """Return a compact display range for validation count columns."""
    values = list(values)
    if not values:
        return "none"
    low = min(values)
    high = max(values)
    if low == high:
        return str(low)
    return f"{low}..{high}"
