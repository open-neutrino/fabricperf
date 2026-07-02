"""Latency-mode analyzer for FabricPerf trace output."""

from __future__ import annotations

import statistics
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neutrino import TraceHeader, TraceSection

from .common import (
    count_range,
    format_float,
    iter_trace_dirs,
    load_reader,
    parse_rank,
    resolve_trace_root,
)

# Latency probe sentinel for samples that matched the mailbox too early.
LATENCY_INVALID_SENTINEL = (1 << 63) - 1
# New latency probes keep raw signed latency and set this channel control bit.
LATENCY_INVALID_FLAG = 1 << 31
# Receive hook packs srcPeer + 1 into high vblock bits for pair attribution.
LATENCY_SOURCE_SHIFT = 16
LATENCY_CHANNEL_MASK = (1 << LATENCY_SOURCE_SHIFT) - 1
LATENCY_SOURCE_UNKNOWN = -1
# NVIDIA %globaltimer is recorded in nanosecond ticks on supported GPUs.
GLOBALTIMER_TICKS_PER_US = 1000.0


def format_latency_us(ticks: float) -> str:
    """Return a display value in microseconds for one globaltimer delta."""
    return format_float(float(ticks) / GLOBALTIMER_TICKS_PER_US)


@dataclass
class ChannelStats:
    """Accumulate latency samples for one channel.

    Motivation: per-channel latency summaries need both values and step ranges.
    Example: channel 0 collects all sr_latency records with vblock or cta 0.
    """

    latencies: list[int] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    invalid_steps: list[int] = field(default_factory=list)

    def add(self, latency: int, step: int) -> None:
        """Append one latency/step pair for later summary statistics.

        Motivation: keep record ingestion simple at the call site. Example:
        add(1200, 64) contributes one sample and extends the displayed step range.
        """
        self.latencies.append(latency)
        self.steps.append(step)

    def add_invalid(self, step: int) -> None:
        """Record one invalid early-hit sample without polluting latency stats.

        Motivation: the probe writes INT64_MAX when a mailbox match needed at
        most one poll miss. Example: step 42 contributes to invalid percentage
        but not to avg/min/p50/max latency.
        """
        self.invalid_steps.append(step)


@dataclass
class TraceStats:
    """Hold all parsed latency records for one per-rank trace directory.

    Motivation: FabricPerf traces are split by rank but printed as one table.
    Example: rank 1 of a two-rank run records its trace name, channel field, and
    per-result-file sample counts.
    """

    trace_name: str
    rank: int
    channel_field: str
    channels: dict[tuple[int, int], ChannelStats] = field(
        default_factory=lambda: defaultdict(ChannelStats)
    )
    per_file_counts: list[dict[tuple[int, int], int]] = field(default_factory=list)
    files: int = 0
    skipped_malformed: int = 0


def parse_compact_sr_latency_section(reader: Any, result_file: Path, section_index: int) -> list[Any]:
    """Parse compact DeepEP latency rows as flat `sr_latency` records.

    Motivation: DeepEP specializes the latency probe down to one map, but the
    trace root may still contain a reader generated from the default three-map
    latency probe. Example: one/two-section results are decoded with the
    generated `sr_latency` namedtuple by reading the first slot in each
    warp-owned stream.
    """
    sr_latency_type = getattr(reader, "sr_latency")
    unpack_latency = struct.Struct("qII").unpack
    compact_records: list[Any] = []
    with result_file.open("rb") as f:
        header = TraceHeader(*struct.unpack("iiiiiiii", f.read(32)))
        sections = [
            TraceSection(*struct.unpack("IIQ", f.read(16)))
            for _ in range(header.numProbes)
        ]
        if len(sections) <= section_index:
            raise ValueError(f"expected compact sr_latency section in {result_file}")

        section = sections[section_index]
        grid_size = header.gridDimX * header.gridDimY * header.gridDimZ
        block_size = header.blockDimX * header.blockDimY * header.blockDimZ
        warp_streams = block_size // section.warpDiv
        for block_idx in range(grid_size):
            for warp_idx in range(warp_streams):
                # Step: DeepEP local latency emits one CTA-thread-zero SAVE, so
                # the first slot carries the sample and the rest of the stream is
                # fixed-capacity zero padding.
                stream_offset = section.offset + (
                    (block_idx * warp_streams + warp_idx) * section.size
                )
                f.seek(stream_offset)
                payload = f.read(16)
                if len(payload) != 16:
                    raise ValueError(f"short compact sr_latency row in {result_file}")
                latency, vblock, step = unpack_latency(payload)
                if latency == 0 and vblock == 0 and step == 0:
                    continue
                compact_records.append(sr_latency_type(latency, vblock, step))
    return compact_records


def parse_sr_latency_records(reader: Any, result_file: Path) -> list[Any]:
    """Return `sr_latency` records from a normal or compact result file."""
    with result_file.open("rb") as f:
        header = TraceHeader(*struct.unpack("iiiiiiii", f.read(32)))
    if header.numProbes in (1, 2):
        # Step: compact DeepEP latency emits only the sr_latency map, while the
        # runtime exchange variant emits sr_latency plus sr_latency_debug.
        # OLD: reader.parse expanded stale three-map schemas and became slow.
        return parse_compact_sr_latency_section(reader, result_file, 0)

    try:
        _header, _sections, records = reader.parse(str(result_file))
        return [
            record
            for block in records["sr_latency"]
            for lane in block
            for record in lane
        ]
    except (IndexError, KeyError):
        # OLD: latency analysis trusted read.py to match every result file.
        # Step: compact DeepEP probes can emit only the sr_latency section.
        return parse_single_section_sr_latency(reader, result_file)


def channel_field_for(record: Any, trace_name: str) -> str:
    """Return the sr_latency grouping field used by one generated record."""
    fields = set(getattr(record, "_fields", ()))
    missing = {"latency", "step"} - fields
    if missing:
        raise ValueError(f"sr_latency records in {trace_name} are missing {sorted(missing)}")
    if "vblock" in fields:
        return "vblock"
    if "cta" in fields:
        return "cta"
    raise ValueError(f"sr_latency records in {trace_name} have neither vblock nor cta")


def is_unwritten_latency_record(record: Any, channel_field: str) -> bool:
    """Return whether one generated sr_latency record is an unwritten zero slot.

    Motivation: Neutrino fixed-size maps are zero-filled, so analyzers must not
    count default capacity as samples. Example: latency 0, step 0, vblock 0 is
    an empty slot, not a measured NVLink latency.
    """
    return (
        int(record.latency) == 0
        and int(record.step) == 0
        and int(getattr(record, channel_field)) == 0
    )


def is_invalid_latency_record(record: Any) -> bool:
    """Return whether one sr_latency record is explicitly marked invalid.

    Motivation: old traces used INT64_MAX and new traces preserve signed latency
    while marking bit 31 of the packed vblock word. Example: invalid rows can be
    filtered without losing the underlying measured value.
    """
    if int(record.latency) == LATENCY_INVALID_SENTINEL:
        return True
    if hasattr(record, "vblock"):
        return (int(record.vblock) & LATENCY_INVALID_FLAG) != 0
    return False


def split_latency_channel(record: Any, channel_field: str) -> tuple[int, int]:
    """Return `(src_peer, channel)` for one latency record.

    Motivation: new latency probes pack `srcPeer + 1` into high vblock bits
    without changing the generated `sr_latency` schema. Example:
    `((2 + 1) << 16) | 7` becomes source 2, vblock 7; older traces return
    source -1.
    """
    raw_channel = int(getattr(record, channel_field))
    if channel_field == "vblock":
        raw_channel &= ~LATENCY_INVALID_FLAG
    if channel_field == "vblock" and raw_channel >= (1 << LATENCY_SOURCE_SHIFT):
        return (
            (raw_channel >> LATENCY_SOURCE_SHIFT) - 1,
            raw_channel & LATENCY_CHANNEL_MASK,
        )
    return LATENCY_SOURCE_UNKNOWN, raw_channel


def read_trace(trace_name: str, trace_dir: Path) -> TraceStats:
    """Read all latency records for one trace directory into TraceStats."""
    reader = load_reader(trace_dir)
    result_files = sorted((trace_dir / "result").glob("*.bin"))
    if not result_files:
        raise FileNotFoundError(f"no result/*.bin files under: {trace_dir}")
    rank = parse_rank(trace_dir)

    stats: TraceStats | None = None
    skipped_malformed = 0
    for result_file in result_files:
        try:
            sr_sections = parse_sr_latency_records(reader, result_file)
        except (OSError, struct.error, ValueError):
            skipped_malformed += 1
            continue
        sr_records = sr_sections
        if not sr_records:
            continue

        channel_field = channel_field_for(sr_records[0], trace_name)
        if stats is not None and stats.channel_field != channel_field:
            raise ValueError(
                f"channel field changed for {trace_dir}: {stats.channel_field} to {channel_field}"
            )

        file_counts: dict[tuple[int, int], int] = defaultdict(int)
        for record in sr_records:
            if is_unwritten_latency_record(record, channel_field):
                continue
            if stats is None:
                stats = TraceStats(trace_name=trace_name, rank=rank, channel_field=channel_field)
            step = int(record.step)
            channel = split_latency_channel(record, channel_field)
            if is_invalid_latency_record(record):
                stats.channels[channel].add_invalid(step)
                file_counts[channel] += 1
                continue
            stats.channels[channel].add(int(record.latency), step)
            file_counts[channel] += 1
        if stats is not None and file_counts:
            stats.files += 1
            stats.per_file_counts.append(dict(file_counts))

    if stats is None:
        raise ValueError(f"no sr_latency records found under: {trace_dir}")
    stats.skipped_malformed = skipped_malformed
    return stats


def rows_for_stats(stats: TraceStats) -> list[list[str]]:
    """Build display rows for one trace's per-channel latency statistics."""
    rows: list[list[str]] = []
    for src_peer, channel in sorted(stats.channels):
        channel_stats = stats.channels[(src_peer, channel)]
        latencies = channel_stats.latencies
        steps = channel_stats.steps
        invalid = len(channel_stats.invalid_steps)
        total = len(latencies) + invalid
        if total == 0:
            continue
        all_steps = steps + channel_stats.invalid_steps
        invalid_pct = (100.0 * invalid / total) if total > 0 else 0.0
        avg = format_latency_us(statistics.fmean(latencies)) if latencies else "n/a"
        p50 = format_latency_us(float(statistics.median(latencies))) if latencies else "n/a"
        rows.append(
            [
                stats.trace_name,
                str(stats.rank),
                str(src_peer) if src_peer >= 0 else "n/a",
                stats.channel_field,
                str(channel),
                str(len(latencies)),
                str(invalid),
                format_float(invalid_pct),
                avg,
                format_latency_us(min(latencies)) if latencies else "n/a",
                p50,
                format_latency_us(max(latencies)) if latencies else "n/a",
                f"{min(all_steps)}..{max(all_steps)}",
            ]
        )
    return rows


def print_table(rows: list[list[str]]) -> None:
    """Print the public latency summary table."""
    headers = [
        "trace",
        "rank",
        "src",
        "channel",
        "id",
        "count",
        "invalid",
        "invalid_pct",
        "avg_us",
        "min_us",
        "p50_us",
        "max_us",
        "steps",
    ]
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.rjust(width) for cell, width in zip(row, widths)))


def print_validation(stats_by_trace: list[TraceStats]) -> None:
    """Print the optional per-rank latency validation table."""
    headers = [
        "trace",
        "rank",
        "files",
        "channels",
        "per_file",
        "total",
        "invalid",
        "malformed",
    ]
    rows: list[list[str]] = []
    for stats in sorted(stats_by_trace, key=lambda item: (item.trace_name, item.rank)):
        channels = sorted(stats.channels)
        per_file_counts = [
            count for file_counts in stats.per_file_counts for count in file_counts.values()
        ]
        total_counts = [len(stats.channels[channel].latencies) for channel in channels]
        invalid_count = sum(len(stats.channels[channel].invalid_steps) for channel in channels)
        rows.append(
            [
                stats.trace_name,
                str(stats.rank),
                str(stats.files),
                str(len(channels)),
                count_range(per_file_counts),
                count_range(total_counts),
                str(invalid_count),
                str(stats.skipped_malformed),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    print()
    print("validation")
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.rjust(width) for cell, width in zip(row, widths)))


def print_malformed_summary(stats_by_trace: list[TraceStats]) -> None:
    """Print skipped-record diagnostics for traces that had malformed rows."""
    malformed = [stats for stats in stats_by_trace if stats.skipped_malformed > 0]
    if not malformed:
        return

    print("", file=sys.stderr)
    for stats in malformed:
        print(
            f"{stats.trace_name} rank {stats.rank}: skipped "
            f"{stats.skipped_malformed} malformed tag/step records",
            file=sys.stderr,
        )


def analyze_latency_roots(raw_roots: list[str], validate: bool = False) -> int:
    """Analyze latency traces for all requested roots and print summaries."""
    stats_by_trace: list[TraceStats] = []
    for raw_root in raw_roots:
        root = resolve_trace_root(raw_root)
        for trace_name, trace_dir in iter_trace_dirs(root):
            try:
                stats_by_trace.append(read_trace(trace_name, trace_dir))
            except FileNotFoundError as exc:
                # Step: torch.spawn parent traces have read.py/probe.toml but no result files.
                if "no result/*.bin files" in str(exc):
                    continue
                raise
            except ValueError as exc:
                # Step: some launcher/helper traces have results but no latency map rows.
                if "no sr_latency records found" in str(exc):
                    continue
                raise

    if not stats_by_trace:
        raise ValueError("no sr_latency records found under requested trace roots")

    rows: list[list[str]] = []
    for stats in sorted(stats_by_trace, key=lambda item: (item.trace_name, item.rank)):
        rows.extend(rows_for_stats(stats))

    print_table(rows)
    if validate:
        print_validation(stats_by_trace)
    print_malformed_summary(stats_by_trace)
    return 0
