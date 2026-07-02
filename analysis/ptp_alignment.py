"""PTP offset closure analyzer for FabricPerf latency traces."""

from __future__ import annotations

import argparse
import math
import statistics
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from neutrino import TraceHeader, TraceSection

from .common import iter_trace_dirs, load_reader, parse_rank, resolve_trace_root


DEFAULT_SAMPLES_PER_PAIR = 20
DEFAULT_COMPACT_DSTS = 7
DEFAULT_MAX_DEVICES = 8
DEFAULT_ROOT_RANK = 0


@dataclass(frozen=True)
class PtpRecord:
    """One decoded PTP offset sample.

    Motivation: `ptp_metrics` stores compact slots, not explicit pair ids.
    Example: slot `(0 * 7 + 1) * 20 + 3` becomes source 0, destination 2,
    sample index 3 after compact-destination expansion.
    """

    trace: str
    rank: int
    file_index: int
    slot_index: int
    sample_index: int
    src: int
    dst: int
    offset: int
    latency: int


@dataclass(frozen=True)
class Summary:
    """Summary statistics for one residual distribution.

    Motivation: closure quality is judged by signed residuals and their spread.
    Example: `O[0,2] - (O[0,1] + O[1,2])` should have small mean and stdev.
    """

    count: int
    mean: float
    stdev: float
    min_value: int
    p50: float
    p95: float
    p99: float
    max_value: int
    max_abs: int


def flatten_records(value: Any) -> Iterable[Any]:
    """Yield generated Neutrino records from nested map output.

    Motivation: generated readers return block/warp/array nesting. Example:
    `flatten_records(records["ptp_metrics"])` yields each namedtuple row.
    """
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        yield value
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten_records(item)
        return
    yield value


def percentile(sorted_values: list[int], pct: float) -> float:
    """Return an interpolated percentile from sorted integer values."""
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * pct / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(sorted_values[low])
    fraction = position - low
    return sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction


def summarize(values: list[int]) -> Summary:
    """Build signed residual summary statistics.

    Motivation: PTP alignment errors are signed clock-domain residuals. Example:
    a residual list of `[-2, 1, 4]` keeps the negative value in the mean and
    uses `max_abs` only as an outlier ranking helper.
    """
    ordered = sorted(values)
    return Summary(
        count=len(values),
        mean=statistics.fmean(values),
        stdev=statistics.pstdev(values) if len(values) > 1 else 0.0,
        min_value=ordered[0],
        p50=percentile(ordered, 50),
        p95=percentile(ordered, 95),
        p99=percentile(ordered, 99),
        max_value=ordered[-1],
        max_abs=max(abs(value) for value in values),
    )


def format_float(value: float) -> str:
    """Format a float compactly for table output."""
    if math.isnan(value):
        return "n/a"
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}"


def decode_ptp_slot(
    slot_index: int,
    samples_per_pair: int = DEFAULT_SAMPLES_PER_PAIR,
    compact_dsts: int = DEFAULT_COMPACT_DSTS,
) -> tuple[int, int, int]:
    """Decode one compact `ptp_metrics` slot into `(src, dst, sample)`.

    Motivation: the PTX stores ordered pairs as `src * 7 + dst_without_src`.
    Example: with eight-rank layout, pair index `1 * 7 + 2` means `1 -> 3`.
    """
    pair_index = slot_index // samples_per_pair
    sample_index = slot_index % samples_per_pair
    src = pair_index // compact_dsts
    compact_dst = pair_index % compact_dsts
    dst = compact_dst if compact_dst < src else compact_dst + 1
    return src, dst, sample_index


def is_unwritten_ptp_record(record: Any) -> bool:
    """Return whether a `ptp_metrics` row is an unused zero-filled slot."""
    return int(record.offset) == 0 and int(record.latency) == 0


def parse_compact_ptp_records(reader: Any, result_file: Path, section_index: int) -> list[Any]:
    """Decode DeepEP compact PTP rows from one binary result file.

    Motivation: DeepEP PTP traces keep `sr_latency` as section 0 and append
    `ptp_metrics` as section 1, while the generated reader may still expect the
    generic latency probe order. Example: a two-section DeepEP result is parsed
    by reading section 1 directly as `(offset, latency)` records.
    """
    ptp_metrics_type = getattr(reader, "ptp_metrics")
    unpack_ptp = struct.Struct("qq").unpack
    compact_records: list[Any] = []
    with result_file.open("rb") as f:
        header = TraceHeader(*struct.unpack("iiiiiiii", f.read(32)))
        sections = [
            TraceSection(*struct.unpack("IIQ", f.read(16)))
            for _ in range(header.numProbes)
        ]
        if len(sections) <= section_index:
            raise ValueError(f"expected compact ptp_metrics section in {result_file}")

        section = sections[section_index]
        # Step: the PTP save loop is gated to CTA 0, warp 0, lane 0, so useful
        # rows live in the first warp-owned stream; the rest is zero padding.
        f.seek(section.offset)
        payload = f.read(section.size)
        if len(payload) != section.size:
            raise ValueError(f"short compact ptp_metrics section in {result_file}")

    for offset in range(0, len(payload), 16):
        compact_records.append(ptp_metrics_type(*unpack_ptp(payload[offset:offset + 16])))
    return compact_records


def parse_ptp_records(reader: Any, result_file: Path) -> list[Any]:
    """Return flat `ptp_metrics` records from generic or compact traces.

    Motivation: the generic reader is correct for normal latency traces, but
    compact DeepEP PTP results intentionally use a different map order. Example:
    header.numProbes == 2 reads section 1 directly instead of expanding stale
    `read.py` section names.
    """
    with result_file.open("rb") as f:
        header = TraceHeader(*struct.unpack("iiiiiiii", f.read(32)))
    if header.numProbes == 2:
        return parse_compact_ptp_records(reader, result_file, 1)
    if header.numProbes == 1:
        return []

    _header, _sections, records = reader.parse(str(result_file))
    if "ptp_metrics" not in records:
        return []
    return list(flatten_records(records["ptp_metrics"]))


def read_ptp_records(
    trace_name: str,
    trace_dir: Path,
    devices: int,
    samples_per_pair: int,
    compact_dsts: int,
) -> list[PtpRecord]:
    """Read decoded PTP records from one rank trace directory.

    Motivation: only the destination rank writes useful rows for its local
    offset table. Example: rank 2 should emit rows with decoded `dst == 2`.
    """
    rank = parse_rank(trace_dir)
    reader = load_reader(trace_dir)
    records_out: list[PtpRecord] = []
    result_files = sorted((trace_dir / "result").glob("*.bin"))
    for file_index, result_file in enumerate(result_files):
        for slot_index, record in enumerate(parse_ptp_records(reader, result_file)):
            if is_unwritten_ptp_record(record):
                continue
            src, dst, sample_index = decode_ptp_slot(slot_index, samples_per_pair, compact_dsts)
            if src >= devices or dst >= devices or src == dst:
                continue
            records_out.append(
                PtpRecord(
                    trace=trace_name,
                    rank=rank,
                    file_index=file_index,
                    slot_index=slot_index,
                    sample_index=sample_index,
                    src=src,
                    dst=dst,
                    offset=int(record.offset),
                    latency=int(record.latency),
                )
            )
    return records_out


def infer_devices(trace_root: Path) -> int:
    """Infer device count from trace ranks, capped at FabricPerf's PTP layout."""
    ranks = [parse_rank(trace_dir) for _trace_name, trace_dir in iter_trace_dirs(trace_root)]
    if not ranks:
        raise ValueError(f"no trace ranks found under {trace_root}")
    return min(max(ranks) + 1, DEFAULT_MAX_DEVICES)


def read_pair_offsets(
    trace_root: Path,
    devices: int | None,
    samples_per_pair: int,
    compact_dsts: int,
) -> dict[tuple[int, int], list[int]]:
    """Read all PTP offsets grouped by ordered `(src, dst)` pair."""
    actual_devices = infer_devices(trace_root) if devices is None else devices
    grouped: dict[tuple[int, int], list[tuple[int, int, int, int]]] = defaultdict(list)
    for trace_name, trace_dir in iter_trace_dirs(trace_root):
        try:
            records = read_ptp_records(
                trace_name,
                trace_dir,
                actual_devices,
                samples_per_pair,
                compact_dsts,
            )
        except ValueError as exc:
            # Step: torch.spawn/helper traces can have read.py/result files but
            # no rank metadata; they do not carry useful PTP rows.
            if "could not infer rank" in str(exc):
                continue
            raise
        for record in records:
            # Sort key aligns repeated launches and sample indices across pairs.
            grouped[(record.src, record.dst)].append(
                (record.file_index, record.sample_index, record.slot_index, record.offset)
            )
    return {
        pair: [offset for _file_index, _sample_index, _slot_index, offset in sorted(values)]
        for pair, values in grouped.items()
    }


def derive_root_relative_offsets(
    pair_offsets: dict[tuple[int, int], list[int]],
    devices: int,
    root: int = DEFAULT_ROOT_RANK,
) -> dict[tuple[int, int], list[int]]:
    """Derive ordered pair offsets from root-relative PTP samples.

    Functionality: the simplified PTP path measures `root -> rank`; any
    `src -> dst` offset is then `R_dst - R_src`. Example: if `O[0,1] = 10`
    and `O[0,2] = 30`, the derived `O[1,2]` is `20`.
    """
    root_series: dict[int, list[int]] = {}
    for rank in range(devices):
        if rank == root:
            continue
        direct = pair_offsets.get((root, rank), [])
        if direct:
            root_series[rank] = list(direct)
            continue
        inverse = pair_offsets.get((rank, root), [])
        if inverse:
            root_series[rank] = [-value for value in inverse]

    max_samples = max((len(values) for values in root_series.values()), default=0)
    root_series[root] = [0] * max_samples

    derived: dict[tuple[int, int], list[int]] = {}
    for src in range(devices):
        for dst in range(devices):
            if src == dst or src not in root_series or dst not in root_series:
                continue
            count = min(len(root_series[src]), len(root_series[dst]))
            if count == 0:
                continue
            derived[(src, dst)] = [
                root_series[dst][index] - root_series[src][index]
                for index in range(count)
            ]
    return derived


def closure_residuals(
    pair_offsets: dict[tuple[int, int], list[int]],
    src: int,
    mid: int,
    dst: int,
) -> list[int]:
    """Return `O[src,dst] - (O[src,mid] + O[mid,dst])` residuals."""
    direct = pair_offsets.get((src, dst), [])
    first_leg = pair_offsets.get((src, mid), [])
    second_leg = pair_offsets.get((mid, dst), [])
    count = min(len(direct), len(first_leg), len(second_leg))
    return [
        direct[index] - (first_leg[index] + second_leg[index])
        for index in range(count)
    ]


def reciprocal_residuals(
    pair_offsets: dict[tuple[int, int], list[int]],
    first: int,
    second: int,
) -> list[int]:
    """Return `O[first,second] + O[second,first]` reciprocal residuals."""
    forward = pair_offsets.get((first, second), [])
    backward = pair_offsets.get((second, first), [])
    count = min(len(forward), len(backward))
    return [forward[index] + backward[index] for index in range(count)]


def all_closure_summaries(
    pair_offsets: dict[tuple[int, int], list[int]],
    devices: int,
) -> list[tuple[int, int, int, Summary]]:
    """Build closure summaries for all ordered distinct triples."""
    rows: list[tuple[int, int, int, Summary]] = []
    for src in range(devices):
        for mid in range(devices):
            for dst in range(devices):
                if src == mid or mid == dst or src == dst:
                    continue
                residuals = closure_residuals(pair_offsets, src, mid, dst)
                if residuals:
                    rows.append((src, mid, dst, summarize(residuals)))
    return rows


def all_reciprocal_summaries(
    pair_offsets: dict[tuple[int, int], list[int]],
    devices: int,
) -> list[tuple[int, int, Summary]]:
    """Build reciprocal summaries for all unordered pairs."""
    rows: list[tuple[int, int, Summary]] = []
    for first in range(devices):
        for second in range(first + 1, devices):
            residuals = reciprocal_residuals(pair_offsets, first, second)
            if residuals:
                rows.append((first, second, summarize(residuals)))
    return rows


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print one aligned table to stdout."""
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.rjust(width) for cell, width in zip(row, widths)))


def summary_cells(summary: Summary) -> list[str]:
    """Convert one Summary to common table cells."""
    return [
        str(summary.count),
        format_float(summary.mean),
        format_float(summary.stdev),
        str(summary.min_value),
        format_float(summary.p50),
        format_float(summary.p95),
        format_float(summary.p99),
        str(summary.max_value),
        str(summary.max_abs),
    ]


def analyze_trace_root(args: argparse.Namespace, raw_root: str) -> None:
    """Analyze one FabricPerf trace root and print closure diagnostics."""
    trace_root = resolve_trace_root(raw_root)
    devices = infer_devices(trace_root) if args.devices is None else args.devices
    pair_offsets = read_pair_offsets(
        trace_root,
        devices,
        args.samples_per_pair,
        args.compact_dsts,
    )
    all_offsets = [offset for offsets in pair_offsets.values() for offset in offsets]
    closure_rows = all_closure_summaries(pair_offsets, devices)
    reciprocal_rows = all_reciprocal_summaries(pair_offsets, devices)
    closure_values = [
        value
        for src, mid, dst, _summary in closure_rows
        for value in closure_residuals(pair_offsets, src, mid, dst)
    ]
    reciprocal_values = [
        value
        for first, second, _summary in reciprocal_rows
        for value in reciprocal_residuals(pair_offsets, first, second)
    ]

    print()
    print(f"trace_root {trace_root}")
    print(f"devices {devices}")
    print(f"pairs_with_offsets {len(pair_offsets)}")
    print(f"offset_samples {len(all_offsets)}")

    if closure_values:
        print("\nclosure_global O[src,dst] - (O[src,mid] + O[mid,dst])")
        print_table(
            ["count", "mean", "stdev", "min", "p50", "p95", "p99", "max", "max_abs"],
            [summary_cells(summarize(closure_values))],
        )
    if reciprocal_values:
        print("\nreciprocal_global O[a,b] + O[b,a]")
        print_table(
            ["count", "mean", "stdev", "min", "p50", "p95", "p99", "max", "max_abs"],
            [summary_cells(summarize(reciprocal_values))],
        )

    closure_rows.sort(key=lambda row: (row[3].stdev, row[3].max_abs), reverse=True)
    print(f"\ntop_closure_triples_by_stdev top={args.top}")
    print_table(
        ["src", "mid", "dst", "count", "mean", "stdev", "min", "p50", "p95", "p99", "max", "max_abs"],
        [
            [str(src), str(mid), str(dst), *summary_cells(summary)]
            for src, mid, dst, summary in closure_rows[: args.top]
        ],
    )

    reciprocal_rows.sort(key=lambda row: (row[2].stdev, row[2].max_abs), reverse=True)
    print(f"\ntop_reciprocal_pairs_by_stdev top={args.top}")
    print_table(
        ["a", "b", "count", "mean", "stdev", "min", "p50", "p95", "p99", "max", "max_abs"],
        [
            [str(first), str(second), *summary_cells(summary)]
            for first, second, summary in reciprocal_rows[: args.top]
        ],
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for PTP closure analysis."""
    parser = argparse.ArgumentParser(
        description="Check FabricPerf PTP offset closure across ordered GPU triples.",
    )
    parser.add_argument(
        "trace_roots",
        nargs="+",
        help="FabricPerf trace root(s), usually one collective run directory.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Device count. Defaults to max rank + 1 inferred from each trace root.",
    )
    parser.add_argument(
        "--samples-per-pair",
        type=int,
        default=DEFAULT_SAMPLES_PER_PAIR,
        help="PTP samples per ordered pair in latency.probe.",
    )
    parser.add_argument(
        "--compact-dsts",
        type=int,
        default=DEFAULT_COMPACT_DSTS,
        help="Compact destination slots per source in ptp_metrics.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=12,
        help="Rows to show in worst-triple and worst-pair tables.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run PTP closure analysis for one or more trace roots."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    for raw_root in args.trace_roots:
        analyze_trace_root(args, raw_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
