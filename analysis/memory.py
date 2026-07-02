"""Memory-mode analyzer for FabricPerf CUPTI and probe CSV output."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import (
    bps_from_bytes,
    float_field,
    format_raw_metrics,
    iter_trace_dirs,
    load_reader,
    number_field,
    parse_rank,
    parse_raw_metrics,
    read_csv_rows,
    resolve_trace_root,
    write_long_metrics_csv,
    write_wide_csv,
)


@dataclass
class ProbeByteRow:
    """Accumulate one launch's probe-backed memory byte totals.

    Motivation: FabricPerf one-pass mode stores per-thread byte counters, but
    the CSV needs one launch-level row. Example: launch 0 sums all
    ld_global_bytes records into probe_read_bytes.
    """

    launch_index: int
    ld_global_bytes: int = 0
    st_global_bytes: int = 0
    cp_async_bytes: int = 0
    records: int = 0

    def add(self, record: Any) -> None:
        """Add one generated-reader record to the launch total.

        Motivation: the reader exposes typed fields but no aggregate helper.
        Example: one thread record contributes its ld/st/cp.async byte counts.
        """
        self.ld_global_bytes += int(record.ld_global_bytes)
        self.st_global_bytes += int(record.st_global_bytes)
        self.cp_async_bytes += int(record.cp_async_bytes)
        self.records += 1

    @property
    def read_bytes(self) -> int:
        """Return probe-backed read-side bytes for existing read columns.

        Motivation: cp.async reads from global memory even though it is not an
        ld.global instruction. Example: 32 ld bytes plus 16 async bytes reports
        48 read bytes.
        """
        return self.ld_global_bytes + self.cp_async_bytes

    @property
    def write_bytes(self) -> int:
        """Return probe-backed write-side bytes for existing write columns.

        Motivation: st.global instructions are the probe-backed write source.
        Example: 64 st.global bytes reports 64 write bytes.
        """
        return self.st_global_bytes


@dataclass
class ProbeByteStats:
    """Hold one trace directory's probe-backed byte totals by launch index.

    Motivation: CUPTI PM rows and Neutrino result files meet at launch_index in
    one-pass mode. Example: PM row launch_index=0 receives probe bytes from the
    same launch_index record group.
    """

    trace_dir: Path
    by_launch: dict[int, ProbeByteRow] = field(default_factory=dict)

    def add(self, record: Any) -> None:
        """Add one memory byte record to its launch bucket.

        Motivation: every thread saves a record, so grouping keeps the merge
        independent of block size. Example: all launch_index=2 records sum into
        one ProbeByteRow.
        """
        launch_index = int(record.launch_index)
        if launch_index not in self.by_launch:
            self.by_launch[launch_index] = ProbeByteRow(launch_index=launch_index)
        self.by_launch[launch_index].add(record)


def merge_probe_bytes_into_row(row: dict[str, str], probe: ProbeByteRow) -> bool:
    """Overlay one probe byte aggregate into an existing PM Sampling CSV row.

    Motivation: PM Sampling supplies NVLink counters, while the Neutrino probe
    supplies memory/XBAR bytes in the same launch. Example: launch_index=0 gets
    dram_read_Bps from probe read bytes and nvlink_rx_Bps from PM.
    """
    duration_s = float_field(row.get("duration_s"))
    order, raw = parse_raw_metrics(row.get("raw_metrics", ""))
    read_bytes = probe.read_bytes
    write_bytes = probe.write_bytes

    row["dram_read_Bps"] = bps_from_bytes(read_bytes, duration_s)
    row["dram_write_Bps"] = bps_from_bytes(write_bytes, duration_s)
    row["xbar_read_Bps"] = bps_from_bytes(read_bytes, duration_s)
    row["xbar_write_Bps"] = bps_from_bytes(write_bytes, duration_s)
    row["xbar_metric"] = "fabricperf_probe_read_bytes"
    row["xbar_value"] = number_field(read_bytes)

    raw["fabricperf_probe_backend"] = "neutrino_probe"
    raw["probe_merge_key"] = "launch_index"
    raw["probe_records"] = number_field(probe.records)
    raw["probe_ld_global_bytes"] = number_field(probe.ld_global_bytes)
    raw["probe_st_global_bytes"] = number_field(probe.st_global_bytes)
    raw["probe_cp_async_bytes"] = number_field(probe.cp_async_bytes)
    raw["probe_read_bytes"] = number_field(read_bytes)
    raw["probe_write_bytes"] = number_field(write_bytes)
    row["raw_metrics"] = format_raw_metrics(order, raw)
    return True


def read_probe_byte_stats(trace_dir: Path) -> ProbeByteStats | None:
    """Read one trace directory's FabricPerf one-pass byte probe records.

    Motivation: the native PM CSV has NVLink values, while the probe result has
    memory/XBAR byte totals. Example: trace/rank0/result/*.bin supplies
    fabricperf_probe_bytes records keyed by launch_index.
    """
    if not (trace_dir / "read.py").is_file() or not (trace_dir / "result").is_dir():
        return None
    reader = load_reader(trace_dir)
    result_files = sorted((trace_dir / "result").glob("*.bin"))
    if not result_files:
        return None

    stats = ProbeByteStats(trace_dir=trace_dir)
    for result_file in result_files:
        _header, _sections, records = reader.parse(str(result_file))
        byte_records = records.get("fabricperf_probe_bytes")
        if byte_records is None:
            continue
        # Accumulate every generated grid/block/lane probe record into launch totals.
        for block in byte_records:
            for lane in block:
                for record in lane:
                    stats.add(record)
    return stats if stats.by_launch else None


def collect_probe_byte_stats_by_rank(trace_root: Path) -> dict[tuple[int, int], ProbeByteRow]:
    """Return probe byte totals keyed by rank and launch index.

    Motivation: current memory runtime writes one shared root PM CSV, while
    Neutrino result buffers remain per-rank. Example: row rank=1 launch_index=3
    merges with trace rank 1's launch 3 probe aggregate.
    """
    stats_by_rank: dict[tuple[int, int], ProbeByteRow] = {}
    for _trace_name, trace_dir in iter_trace_dirs(trace_root):
        probe_stats = read_probe_byte_stats(trace_dir)
        if probe_stats is None:
            continue
        rank = parse_rank(trace_dir)
        for launch_index, probe in probe_stats.by_launch.items():
            stats_by_rank[(rank, launch_index)] = probe
    return stats_by_rank


def merge_existing_root_memory_csv(trace_root: Path) -> bool:
    """Merge per-rank probe bytes into an already-written root PM CSV.

    Motivation: the plugin writes `trace_root/fabricperf_cupti.csv` directly
    during the run when ranks share a trace root. Example: analyzer reruns after
    MPI exit should not fail just because there are no per-rank PM CSV files.
    """
    csv_path = trace_root / "fabricperf_cupti.csv"
    if not csv_path.is_file():
        return False

    fieldnames, rows = read_csv_rows(csv_path)
    if not rows:
        return False

    probe_by_rank = collect_probe_byte_stats_by_rank(trace_root)
    for row in rows:
        try:
            rank = int(row.get("rank", "0") or "0")
        except ValueError:
            rank = 0
        try:
            launch_index = int(row.get("launch_index", "0") or "0")
        except ValueError:
            launch_index = 0
        probe = probe_by_rank.get((rank, launch_index))
        if probe is not None:
            merge_probe_bytes_into_row(row, probe)

    try:
        write_wide_csv(csv_path, fieldnames, rows)
    except OSError as exc:
        raise ValueError(f"failed to rewrite root FabricPerf CSV {csv_path}: {exc}") from exc
    if write_long_metrics_csv(trace_root, rows) != 0:
        raise ValueError(f"failed to write long-form root FabricPerf CSVs under {trace_root}")
    return True


def merge_one_pass_probe_output(trace_root: Path) -> bool:
    """Merge one-pass probe byte records into PM Sampling CSV output.

    Motivation: normal `--plugin fabricperf` now collects PM NVLink and probe
    memory bytes in one workload execution. Example: per-rank
    fabricperf_cupti.csv files are merged into trace_root/fabricperf_cupti.csv.
    """
    if merge_existing_root_memory_csv(trace_root):
        return True

    merged_rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None

    try:
        trace_items = list(iter_trace_dirs(trace_root))
    except FileNotFoundError:
        return False

    for _trace_name, trace_dir in trace_items:
        csv_path = trace_dir / "fabricperf_cupti.csv"
        if not csv_path.is_file():
            continue
        fields, rows = read_csv_rows(csv_path)
        if fieldnames is None:
            fieldnames = fields
        probe_stats = read_probe_byte_stats(trace_dir)
        for row in rows:
            try:
                launch_index = int(row.get("launch_index", "0") or "0")
            except ValueError:
                launch_index = 0
            probe = probe_stats.by_launch.get(launch_index) if probe_stats is not None else None
            if probe is not None:
                merge_probe_bytes_into_row(row, probe)
            merged_rows.append(row)

    if not merged_rows or fieldnames is None:
        return False

    output_path = trace_root / "fabricperf_cupti.csv"
    try:
        write_wide_csv(output_path, fieldnames, merged_rows)
    except OSError as exc:
        raise ValueError(f"failed to write one-pass FabricPerf CSV {output_path}: {exc}") from exc
    if write_long_metrics_csv(trace_root, merged_rows) != 0:
        raise ValueError(f"failed to write long-form one-pass FabricPerf CSVs under {trace_root}")
    return True


def analyze_memory_roots(raw_roots: list[str]) -> int:
    """Analyze memory-mode outputs for all requested trace roots."""
    merged_any = False
    for raw_root in raw_roots:
        root = resolve_trace_root(raw_root)
        merged_any = merge_one_pass_probe_output(root) or merged_any
    if not merged_any:
        raise ValueError("no FabricPerf memory PM/probe output found")
    return 0
