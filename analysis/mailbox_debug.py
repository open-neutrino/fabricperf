"""Mailbox timestamp-pairing analyzer for FabricPerf latency traces."""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .common import iter_trace_dirs, load_reader, parse_rank, resolve_trace_root


LATENCY_INVALID_SENTINEL = (1 << 63) - 1
LATENCY_INVALID_FLAG = 1 << 31
DEFAULT_BOUND = 1_000_000
DEFAULT_HUGE_BOUND = 1_000_000_000_000
TAG_KIND = 160


@dataclass(frozen=True)
class DebugRecord:
    """One decoded latency mailbox debug row.

    Motivation: `sr_latency_debug` stores compact metadata. Example: `meta`
    packs `poll_count` in high 32 bits and `(srcPeer + 1, vblock)` in low bits.
    """

    trace: str
    rank: int
    src: int
    vblock: int
    poll_count: int
    latency: int
    step: int
    tag: int
    tag_step: int
    tag_run_id: int
    tag_kind: int
    sender_time: int
    offset: int
    corrected_sender_time: int
    local_time: int
    invalid: bool


@dataclass(frozen=True)
class PairSummary:
    """Aggregate suspicious mailbox records for one `(src, dst)` pair."""

    total: int
    invalid: int
    positive: int
    negative: int
    huge: int
    bounded_count: int
    bounded_mean: float | None
    bounded_p50: float | None
    min_latency: int | None
    max_latency: int | None


def flatten_records(value: Any) -> Iterable[Any]:
    """Yield generated Neutrino records from nested map output."""
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        yield value
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten_records(item)
        return
    yield value


def is_unwritten_debug_record(record: Any) -> bool:
    """Return whether one debug row is an unused zero-filled slot."""
    return all(int(getattr(record, field)) == 0 for field in record._fields)


def decode_meta(meta: int) -> tuple[int, int, int]:
    """Decode `(poll_count, srcPeer, vblock)` from a debug metadata word."""
    poll_count = (meta >> 32) & 0xFFFFFFFF
    packed = (meta & 0xFFFFFFFF) & ~LATENCY_INVALID_FLAG
    packed_src = packed >> 16
    src = packed_src - 1 if packed_src > 0 else -1
    vblock = packed & 0xFFFF
    return poll_count, src, vblock


def decode_invalid(meta: int, latency: int) -> bool:
    """Return whether a debug row is invalid in legacy or flag encoding."""
    return latency == LATENCY_INVALID_SENTINEL or (meta & LATENCY_INVALID_FLAG) != 0


def decode_tag(tag: int) -> tuple[int, int, int]:
    """Decode `(step, run_id, kind)` from the active mailbox tag layout."""
    kind = tag & 0xFF
    run_id = (tag >> 8) & 0x00FFFFFF
    step = tag >> 32
    return step, run_id, kind


def read_debug_records(trace_root: Path) -> list[DebugRecord]:
    """Read all `sr_latency_debug` records under one trace root."""
    out: list[DebugRecord] = []
    for trace_name, trace_dir in iter_trace_dirs(trace_root):
        rank = parse_rank(trace_dir)
        reader = load_reader(trace_dir)
        for result_file in sorted((trace_dir / "result").glob("*.bin")):
            _header, _sections, records = reader.parse(str(result_file))
            if "sr_latency_debug" not in records:
                continue
            for record in flatten_records(records["sr_latency_debug"]):
                if is_unwritten_debug_record(record):
                    continue
                poll_count, src, vblock = decode_meta(int(record.meta))
                tag_step, tag_run_id, tag_kind = decode_tag(int(record.tag))
                out.append(
                    DebugRecord(
                        trace=trace_name,
                        rank=rank,
                        src=src,
                        vblock=vblock,
                        poll_count=poll_count,
                        latency=int(record.latency),
                        step=int(record.step),
                        tag=int(record.tag),
                        tag_step=tag_step,
                        tag_run_id=tag_run_id,
                        tag_kind=tag_kind,
                        sender_time=int(record.sender_time),
                        offset=int(record.offset),
                        corrected_sender_time=int(record.corrected_sender_time),
                        local_time=int(record.local_time),
                        invalid=decode_invalid(int(record.meta), int(record.latency)),
                    )
                )
    return out


def summarize_pair(values: list[int], invalid: int, bound: int, huge_bound: int) -> PairSummary:
    """Build one signed pair summary for mailbox latency diagnostics."""
    bounded = [value for value in values if -bound <= value <= bound]
    return PairSummary(
        total=len(values) + invalid,
        invalid=invalid,
        positive=sum(1 for value in values if value > bound),
        negative=sum(1 for value in values if value < -bound),
        huge=sum(1 for value in values if value > huge_bound),
        bounded_count=len(bounded),
        bounded_mean=statistics.fmean(bounded) if bounded else None,
        bounded_p50=statistics.median(bounded) if bounded else None,
        min_latency=min(values) if values else None,
        max_latency=max(values) if values else None,
    )


def format_number(value: float | int | None) -> str:
    """Format nullable numeric table cells."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print one aligned diagnostics table."""
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.rjust(width) for cell, width in zip(row, widths)))


def analyze_root(args: argparse.Namespace, raw_root: str) -> None:
    """Analyze one trace root for mailbox timestamp-pairing errors."""
    trace_root = resolve_trace_root(raw_root)
    records = read_debug_records(trace_root)
    print()
    print(f"trace_root {trace_root}")
    print(f"debug_records {len(records)}")
    if not records:
        print("sr_latency_debug map not found or empty")
        return

    valid = [record for record in records if not record.invalid]
    invalid = len(records) - len(valid)
    kind_mismatch = sum(1 for record in records if record.tag_kind != TAG_KIND)
    step_mismatch = sum(1 for record in records if record.tag_step != record.step)
    recompute_mismatch = sum(
        1
        for record in valid
        if record.local_time - record.corrected_sender_time != record.latency
    )
    future_sender = sum(1 for record in valid if record.corrected_sender_time > record.local_time)
    zero_sender = sum(1 for record in records if record.sender_time == 0)
    pos_outliers = sum(1 for record in valid if record.latency > args.bound)
    neg_outliers = sum(1 for record in valid if record.latency < -args.bound)
    huge_outliers = sum(1 for record in valid if record.latency > args.huge_bound)

    print_table(
        [
            "records",
            "valid",
            "invalid",
            "kind_bad",
            "step_bad",
            "recompute_bad",
            "future_sender",
            "zero_sender",
            ">bound",
            "<-bound",
            ">huge",
        ],
        [[
            str(len(records)),
            str(len(valid)),
            str(invalid),
            str(kind_mismatch),
            str(step_mismatch),
            str(recompute_mismatch),
            str(future_sender),
            str(zero_sender),
            str(pos_outliers),
            str(neg_outliers),
            str(huge_outliers),
        ]],
    )

    by_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
    invalid_by_pair: dict[tuple[int, int], int] = defaultdict(int)
    for record in records:
        key = (record.src, record.rank)
        if record.invalid:
            invalid_by_pair[key] += 1
        else:
            by_pair[key].append(record.latency)

    rows: list[list[str]] = []
    for key in sorted(set(by_pair) | set(invalid_by_pair)):
        summary = summarize_pair(by_pair.get(key, []), invalid_by_pair.get(key, 0), args.bound, args.huge_bound)
        src, dst = key
        rows.append([
            f"{src}->{dst}",
            str(summary.total),
            str(summary.invalid),
            str(summary.positive),
            str(summary.huge),
            str(summary.negative),
            str(summary.bounded_count),
            format_number(summary.bounded_mean),
            format_number(summary.bounded_p50),
            format_number(summary.min_latency),
            format_number(summary.max_latency),
        ])
    rows.sort(key=lambda row: (int(row[3]), int(row[4]), int(row[5])), reverse=True)
    print(f"\ntop_pairs_by_positive_outliers top={args.top}")
    print_table(
        ["pair", "total", "invalid", ">bound", ">huge", "<-bound", "bounded", "bounded_avg", "bounded_p50", "min", "max"],
        rows[: args.top],
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for mailbox debug analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze FabricPerf latency mailbox timestamp debug records.",
    )
    parser.add_argument("trace_roots", nargs="+", help="FabricPerf trace root(s).")
    parser.add_argument("--bound", type=int, default=DEFAULT_BOUND, help="Signed latency bound.")
    parser.add_argument("--huge-bound", type=int, default=DEFAULT_HUGE_BOUND, help="Large positive outlier bound.")
    parser.add_argument("--top", type=int, default=12, help="Rows to show in per-pair table.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run mailbox debug analysis for one or more trace roots."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    for raw_root in args.trace_roots:
        analyze_root(args, raw_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
