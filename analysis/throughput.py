"""Throughput-mode analyzer for FabricPerf receive-arrival traces."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import (
    THROUGHPUT_BIN_NS,
    THROUGHPUT_CAPTURE_CAPACITY,
    THROUGHPUT_CELLS_PER_WORKGROUP,
    THROUGHPUT_HEADER_CELLS,
    THROUGHPUT_SLOTS,
    THROUGHPUT_WORKGROUPS,
    format_float,
    iter_trace_dirs,
    load_reader,
    parse_rank,
    resolve_trace_root,
    throughput_partitions,
)


@dataclass
class ThroughputAggregate:
    """Accumulate throughput records for one trace/rank/workgroup.

    Motivation: Neutrino may emit multiple result files for multiple launches,
    while the public summary is grouped by workgroup. Example: rank 0 block 7
    sums captured and dropped arrivals across all result files.
    """

    trace_name: str
    rank: int
    variant: str
    workgroup: int
    captured: int = 0
    dropped: int = 0
    dropped_unknown: bool = False
    duration_ticks: int = 0
    bin_counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def add_launch(
        self,
        deltas: list[int],
        total_arrivals: int | None,
        duration_ticks: int | None,
        capacity: int = THROUGHPUT_CAPTURE_CAPACITY,
    ) -> None:
        """Add one result file's workgroup records to the summary.

        Motivation: metadata supplies exact overflow counts, while older traces
        can still report an overflow marker. Example: total_arrivals=130 with
        capacity=127 contributes dropped=3 when slot 0 is the partition header.
        """
        captured = len(deltas)
        self.captured += captured
        if total_arrivals is not None:
            self.dropped += max(0, total_arrivals - min(total_arrivals, capacity))
        elif captured >= capacity:
            self.dropped_unknown = True

        if duration_ticks is not None and duration_ticks > 0:
            self.duration_ticks += duration_ticks
        elif deltas:
            self.duration_ticks += max(deltas)

        for delta in deltas:
            self.bin_counts[delta // THROUGHPUT_BIN_NS] += 1

    @property
    def dropped_field(self) -> str:
        """Return a display value for exact or inferred dropped arrivals."""
        if self.dropped_unknown and self.dropped == 0:
            return "overflow"
        if self.dropped_unknown:
            return f"{self.dropped}+"
        return str(self.dropped)

    @property
    def avg_pps(self) -> float:
        """Return average arrivals per second from nanosecond-scale deltas."""
        if self.duration_ticks <= 0:
            return 0.0
        return float(self.captured) * 1.0e9 / float(self.duration_ticks)

    @property
    def peak_bin_pps(self) -> float:
        """Return peak binned arrivals per second using the fixed bin width."""
        if not self.bin_counts:
            return 0.0
        return float(max(self.bin_counts.values())) * 1.0e9 / float(THROUGHPUT_BIN_NS)


def first_warp_records(records: Any, workgroup: int) -> list[Any]:
    """Return the first generated warp record list for one logical CTA stream.

    Motivation: throughput uses Neutrino's warp map as an allocation vehicle,
    but stores the logical CTA stream in warp slot zero. Example: records[3][0]
    is CTA 3's 4096-cell receive history.
    """
    try:
        if len(records[workgroup]) == 0:
            return []
        return list(records[workgroup][0])
    except (IndexError, TypeError):
        return []


def warp_record_streams(records: Any, block_index: int) -> list[tuple[int, list[Any]]]:
    """Return all generated warp record streams for one CTA.

    Motivation: selected NCCL devFunc fallback uses one lane-zero writer per
    warp so every writer has a valid Neutrino warp-level map address. Example:
    records[3][7] can hold a SendRecv devFunc stream even when records[3][0]
    is empty.
    """
    try:
        return [
            (warp_index, list(raw_slots))
            for warp_index, raw_slots in enumerate(records[block_index])
            if raw_slots
        ]
    except (IndexError, TypeError):
        return []


def workgroup_base_for_warp(block_index: int, block_count: int, warp_count: int, warp_index: int) -> int:
    """Return the public workgroup-id base for a CTA/warp stream.

    Motivation: legacy traces wrote only warp zero, so their workgroup ids must
    remain `block * 4 + slice`. Example: nonzero warp streams are appended
    after the legacy block range to avoid collisions.
    """
    if warp_index == 0:
        return block_index * THROUGHPUT_WORKGROUPS
    extra_stream = block_index * max(0, warp_count - 1) + (warp_index - 1)
    return (block_count + extra_stream) * THROUGHPUT_WORKGROUPS


def throughput_meta_record(meta_records: Any, workgroup: int) -> Any | None:
    """Return one workgroup metadata record when the trace schema has it.

    Motivation: metadata carries total arrivals and duration separately from
    the saturated public record map. Example: total_arrivals=600 means 88
    dropped records with a 512-slot history.
    """
    if meta_records is None:
        return None
    try:
        if len(meta_records[workgroup]) == 0 or len(meta_records[workgroup][0]) == 0:
            return None
        return meta_records[workgroup][0][0]
    except (IndexError, TypeError):
        return None


def inferred_throughput_records(records: list[Any]) -> list[Any]:
    """Infer populated throughput slots for older traces without metadata.

    Motivation: zero-filled Neutrino result memory has no count field unless the
    metadata map exists. Example: sequence=17 or delta_ticks>0 marks a populated
    fallback record.
    """
    populated = [
        record
        for record in records
        if int(getattr(record, "sequence", 0)) != 0 or int(getattr(record, "delta_ticks", 0)) != 0
    ]
    return populated


def throughput_workgroup_cell_records(
    raw_cells: list[Any],
    block_index: int,
    workgroup_base: int | None = None,
) -> list[tuple[int, list[tuple[int, int]], int, int, int]]:
    """Decode the fixed four-workgroup u32-cell throughput layout.

    Motivation: the trace stores sequence structurally, not as payload. Example:
    cell 2 in a workgroup slice becomes sequence 0 with its value as delta_ticks.
    DeepEP uses a compact four-slice map, so the per-workgroup cell count is
    derived from the trace length rather than hard-coded to 1024.
    """
    decoded: list[tuple[int, list[tuple[int, int]], int, int, int]] = []
    # OLD: required THROUGHPUT_SLOTS exactly, which rejected compact DeepEP maps.
    if len(raw_cells) < THROUGHPUT_WORKGROUPS * THROUGHPUT_HEADER_CELLS:
        return decoded
    cells_per_workgroup = len(raw_cells) // THROUGHPUT_WORKGROUPS
    if cells_per_workgroup <= THROUGHPUT_HEADER_CELLS:
        return decoded

    for workgroup_id in range(THROUGHPUT_WORKGROUPS):
        start = workgroup_id * cells_per_workgroup
        duration_ticks = int(getattr(raw_cells[start], "value", 0))
        total_arrivals = int(getattr(raw_cells[start + 1], "value", 0))
        if total_arrivals <= 0:
            continue

        capacity = cells_per_workgroup - THROUGHPUT_HEADER_CELLS
        limit = min(total_arrivals, capacity)
        accepted: list[tuple[int, int]] = []
        payload_start = start + THROUGHPUT_HEADER_CELLS
        for sequence, record in enumerate(raw_cells[payload_start : payload_start + limit]):
            accepted.append((sequence, int(getattr(record, "value", 0))))

        base = block_index * THROUGHPUT_WORKGROUPS if workgroup_base is None else workgroup_base
        workgroup = base + workgroup_id
        decoded.append((workgroup, accepted, total_arrivals, duration_ticks, capacity))
    return decoded


def throughput_partition_records(
    raw_slots: list[Any],
    block_index: int,
    partitions: int,
    workgroup_base: int | None = None,
) -> list[tuple[int, list[tuple[int, int]], int, int, int]]:
    """Decode new slot-0 throughput headers into per-partition launches.

    Motivation: no-atomic throughput stores count/duration in slot 0 of each
    partition. Example: with four partitions, block 2 partition 3 reports as
    workgroup 11 and uses slots 385..511 for arrivals.
    """
    if not raw_slots:
        return []
    # Legacy traces used the generated record count as the stream size.
    partition_size = len(raw_slots) // partitions
    decoded: list[tuple[int, list[tuple[int, int]], int, int, int]] = []
    for partition in range(partitions):
        start = partition * partition_size
        if start >= len(raw_slots):
            continue
        header = raw_slots[start]
        total_arrivals = int(getattr(header, "sequence", 0))
        duration_ticks = int(getattr(header, "delta_ticks", 0))
        capacity = max(0, partition_size - 1)
        if total_arrivals <= 0:
            continue
        limit = min(total_arrivals, capacity)
        accepted = [
            (int(getattr(record, "sequence", sequence)), int(getattr(record, "delta_ticks", 0)))
            for sequence, record in enumerate(raw_slots[start + 1 : start + 1 + limit])
        ]
        base = block_index * partitions if workgroup_base is None else workgroup_base
        workgroup = base + partition
        decoded.append((workgroup, accepted, total_arrivals, duration_ticks, capacity))
    return decoded


def read_throughput_root(root: Path, variant: str) -> tuple[list[ThroughputAggregate], list[dict[str, str]]]:
    """Read throughput arrival records under one trace root.

    Motivation: throughput mode stores timestamp deltas, not pre-binned counters.
    Example: this emits arrival rows for `fabricperf_throughput.csv`.
    """
    aggregates: dict[tuple[str, int, int], ThroughputAggregate] = {}
    csv_rows: list[dict[str, str]] = []
    partitions = throughput_partitions()

    for trace_name, trace_dir in iter_trace_dirs(root):
        reader = load_reader(trace_dir)
        rank = parse_rank(trace_dir)
        result_files = sorted((trace_dir / "result").glob("*.bin"))
        if not result_files:
            continue

        for result_file in result_files:
            _header, _sections, records = reader.parse(str(result_file))
            throughput_records = records.get("fabricperf_throughput")
            if throughput_records is None:
                continue
            meta_records = records.get("fabricperf_throughput_meta")

            block_count = len(throughput_records)
            for block_index in range(block_count):
                streams = warp_record_streams(throughput_records, block_index)
                warp_count = len(throughput_records[block_index]) if block_index < len(throughput_records) else 1
                for warp_index, raw_slots in streams:
                    workgroup_base = workgroup_base_for_warp(block_index, block_count, warp_count, warp_index)
                    launches: list[tuple[int, list[tuple[int, int]], int | None, int | None, int]]
                    if raw_slots and hasattr(raw_slots[0], "value"):
                        launches = throughput_workgroup_cell_records(raw_slots, block_index, workgroup_base)
                    else:
                        meta = throughput_meta_record(meta_records, block_index)
                        if meta is not None and warp_index == 0:
                            total_arrivals = int(meta.total_arrivals)
                            duration_ticks = int(meta.duration_ticks)
                            accepted = [
                                (int(getattr(record, "sequence", sequence)), int(getattr(record, "delta_ticks", 0)))
                                for sequence, record in enumerate(raw_slots[:min(total_arrivals, len(raw_slots))])
                            ]
                            launches = [(workgroup_base, accepted, total_arrivals, duration_ticks, len(raw_slots))]
                        else:
                            # OLD: decode the partitioned two-field layout as a fallback.
                            # Example: older traces used slot 0 as {duration, total}.
                            launches = throughput_partition_records(raw_slots, block_index, partitions, workgroup_base)
                            if not launches:
                                legacy_records = inferred_throughput_records(raw_slots)
                                launches = [(
                                    workgroup_base,
                                    [
                                        (int(getattr(record, "sequence", sequence)), int(getattr(record, "delta_ticks", 0)))
                                        for sequence, record in enumerate(legacy_records)
                                    ],
                                    None,
                                    None,
                                    len(raw_slots),
                                )]

                    for workgroup, accepted, total_arrivals, duration_ticks, capacity in launches:
                        if not accepted and not total_arrivals:
                            continue

                        deltas: list[int] = []
                        for sequence, delta in accepted:
                            deltas.append(delta)
                            csv_rows.append({
                                "trace": trace_name,
                                "rank": str(rank),
                                "variant": variant,
                                "result_file": result_file.name,
                                "workgroup": str(workgroup),
                                "sequence": str(sequence),
                                "delta_ticks": str(delta),
                                "bin_ns": str(THROUGHPUT_BIN_NS),
                                "bin_index": str(delta // THROUGHPUT_BIN_NS),
                            })

                        key = (trace_name, rank, workgroup)
                        if key not in aggregates:
                            aggregates[key] = ThroughputAggregate(
                                trace_name=trace_name,
                                rank=rank,
                                variant=variant,
                                workgroup=workgroup,
                            )
                        aggregates[key].add_launch(deltas, total_arrivals, duration_ticks, capacity)

    return list(aggregates.values()), csv_rows


def write_throughput_csv(root: Path, rows: list[dict[str, str]]) -> None:
    """Write raw FabricPerf throughput records to one CSV file.

    Motivation: downstream analysis needs per-arrival rows, not only the printed
    summary. Example: fabricperf_throughput.csv includes sequence and bin_index
    for every captured receive.
    """
    fieldnames = [
        "trace",
        "rank",
        "variant",
        "result_file",
        "workgroup",
        "sequence",
        "delta_ticks",
        "bin_ns",
        "bin_index",
    ]
    with (root / "fabricperf_throughput.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rows_for_throughput(aggregates: list[ThroughputAggregate]) -> list[list[str]]:
    """Build display rows for throughput workgroup summaries."""
    rows: list[list[str]] = []
    for item in sorted(aggregates, key=lambda value: (value.trace_name, value.rank, value.workgroup)):
        rows.append([
            item.trace_name,
            str(item.rank),
            item.variant,
            str(item.workgroup),
            str(item.captured),
            item.dropped_field,
            str(item.duration_ticks),
            format_float(item.avg_pps),
            format_float(item.peak_bin_pps),
            str(THROUGHPUT_BIN_NS),
        ])
    return rows


def print_throughput_table(rows: list[list[str]]) -> None:
    """Print the public throughput summary table."""
    headers = [
        "trace",
        "rank",
        "variant",
        "workgroup",
        "captured",
        "dropped",
        "duration_ticks",
        "avg_pps",
        "peak_bin_pps",
        "bin_ns",
    ]
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.rjust(width) for cell, width in zip(row, widths)))


def analyze_throughput_roots(raw_roots: list[str], variant: str) -> int:
    """Analyze throughput traces for all requested trace roots."""
    all_aggregates: list[ThroughputAggregate] = []
    wrote_any = False
    for raw_root in raw_roots:
        root = resolve_trace_root(raw_root)
        aggregates, csv_rows = read_throughput_root(root, variant)
        if aggregates:
            wrote_any = True
        write_throughput_csv(root, csv_rows)
        all_aggregates.extend(aggregates)
    if not wrote_any:
        raise ValueError("no FabricPerf throughput output found")
    if all_aggregates:
        print_throughput_table(rows_for_throughput(all_aggregates))
    return 0
