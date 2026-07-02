#!/usr/bin/env python3
"""Dispatch FabricPerf analyzer modes while preserving public imports."""

from __future__ import annotations

import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    from .analysis.common import (
        DEFAULT_TRACE_ROOT,
        DEFAULT_THROUGHPUT_PARTITIONS,
        FABRICPERF_MODE_ENV,
        FABRICPERF_THROUGHPUT_PARTITIONS_ENV,
        LATENCY_MODE,
        MEMORY_MODE,
        REPO_ROOT,
        THIS_FILE,
        THROUGHPUT_BIN_NS,
        THROUGHPUT_CAPTURE_CAPACITY,
        THROUGHPUT_CELLS_PER_WORKGROUP,
        THROUGHPUT_HEADER_CELLS,
        THROUGHPUT_MODE,
        THROUGHPUT_MODES,
        THROUGHPUT_SLOTS,
        THROUGHPUT_WORKGROUPS,
        VALID_MODES,
        VALID_THROUGHPUT_PARTITIONS,
        WORKSPACE_ROOT,
        bps_from_bytes,
        count_range,
        fabricperf_mode,
        fail,
        float_field,
        format_float,
        format_raw_metrics,
        is_diagnostic_raw_metric,
        iter_trace_dirs,
        load_reader,
        number_field,
        parse_args,
        parse_rank,
        parse_raw_metrics,
        read_csv_rows,
        resolve_trace_root,
        throughput_partitions,
        throughput_variant,
        write_long_metrics_csv,
        write_wide_csv,
    )
    from .analysis.latency import (
        ChannelStats,
        TraceStats,
        analyze_latency_roots,
        channel_field_for,
        print_malformed_summary,
        print_table,
        print_validation,
        read_trace,
        rows_for_stats,
    )
    from .analysis.memory import (
        ProbeByteRow,
        ProbeByteStats,
        analyze_memory_roots,
        merge_one_pass_probe_output,
        merge_probe_bytes_into_row,
        read_probe_byte_stats,
    )
    from .analysis.throughput import (
        ThroughputAggregate,
        analyze_throughput_roots,
        first_warp_records,
        inferred_throughput_records,
        print_throughput_table,
        read_throughput_root,
        rows_for_throughput,
        throughput_meta_record,
        throughput_partition_records,
        throughput_workgroup_cell_records,
        write_throughput_csv,
    )
except ImportError:
    from neutrino.plugins.fabricperf.analysis.common import (
        DEFAULT_TRACE_ROOT,
        DEFAULT_THROUGHPUT_PARTITIONS,
        FABRICPERF_MODE_ENV,
        FABRICPERF_THROUGHPUT_PARTITIONS_ENV,
        LATENCY_MODE,
        MEMORY_MODE,
        REPO_ROOT,
        THIS_FILE,
        THROUGHPUT_BIN_NS,
        THROUGHPUT_CAPTURE_CAPACITY,
        THROUGHPUT_CELLS_PER_WORKGROUP,
        THROUGHPUT_HEADER_CELLS,
        THROUGHPUT_MODE,
        THROUGHPUT_MODES,
        THROUGHPUT_SLOTS,
        THROUGHPUT_WORKGROUPS,
        VALID_MODES,
        VALID_THROUGHPUT_PARTITIONS,
        WORKSPACE_ROOT,
        bps_from_bytes,
        count_range,
        fabricperf_mode,
        fail,
        float_field,
        format_float,
        format_raw_metrics,
        is_diagnostic_raw_metric,
        iter_trace_dirs,
        load_reader,
        number_field,
        parse_args,
        parse_rank,
        parse_raw_metrics,
        read_csv_rows,
        resolve_trace_root,
        throughput_partitions,
        throughput_variant,
        write_long_metrics_csv,
        write_wide_csv,
    )
    from neutrino.plugins.fabricperf.analysis.latency import (
        ChannelStats,
        TraceStats,
        analyze_latency_roots,
        channel_field_for,
        print_malformed_summary,
        print_table,
        print_validation,
        read_trace,
        rows_for_stats,
    )
    from neutrino.plugins.fabricperf.analysis.memory import (
        ProbeByteRow,
        ProbeByteStats,
        analyze_memory_roots,
        merge_one_pass_probe_output,
        merge_probe_bytes_into_row,
        read_probe_byte_stats,
    )
    from neutrino.plugins.fabricperf.analysis.throughput import (
        ThroughputAggregate,
        analyze_throughput_roots,
        first_warp_records,
        inferred_throughput_records,
        print_throughput_table,
        read_throughput_root,
        rows_for_throughput,
        throughput_meta_record,
        throughput_partition_records,
        throughput_workgroup_cell_records,
        write_throughput_csv,
    )


__all__ = [
    "ChannelStats",
    "DEFAULT_TRACE_ROOT",
    "DEFAULT_THROUGHPUT_PARTITIONS",
    "FABRICPERF_MODE_ENV",
    "FABRICPERF_THROUGHPUT_PARTITIONS_ENV",
    "LATENCY_MODE",
    "MEMORY_MODE",
    "ProbeByteRow",
    "ProbeByteStats",
    "REPO_ROOT",
    "THIS_FILE",
    "THROUGHPUT_BIN_NS",
    "THROUGHPUT_CAPTURE_CAPACITY",
    "THROUGHPUT_CELLS_PER_WORKGROUP",
    "THROUGHPUT_HEADER_CELLS",
    "THROUGHPUT_MODE",
    "THROUGHPUT_MODES",
    "THROUGHPUT_SLOTS",
    "THROUGHPUT_WORKGROUPS",
    "ThroughputAggregate",
    "TraceStats",
    "VALID_MODES",
    "VALID_THROUGHPUT_PARTITIONS",
    "WORKSPACE_ROOT",
    "analyze_latency_roots",
    "analyze_memory_roots",
    "analyze_throughput_roots",
    "bps_from_bytes",
    "channel_field_for",
    "count_range",
    "fabricperf_mode",
    "fail",
    "first_warp_records",
    "float_field",
    "format_float",
    "format_raw_metrics",
    "inferred_throughput_records",
    "is_diagnostic_raw_metric",
    "iter_trace_dirs",
    "load_reader",
    "main",
    "merge_one_pass_probe_output",
    "merge_probe_bytes_into_row",
    "number_field",
    "parse_args",
    "parse_rank",
    "parse_raw_metrics",
    "print_malformed_summary",
    "print_table",
    "print_throughput_table",
    "print_validation",
    "read_csv_rows",
    "read_probe_byte_stats",
    "read_throughput_root",
    "read_trace",
    "resolve_trace_root",
    "rows_for_stats",
    "rows_for_throughput",
    "throughput_meta_record",
    "throughput_partition_records",
    "throughput_partitions",
    "throughput_variant",
    "throughput_workgroup_cell_records",
    "write_long_metrics_csv",
    "write_throughput_csv",
    "write_wide_csv",
]


def main() -> int:
    """Run the selected FabricPerf analyzer and report trace errors concisely.

    Motivation: automatic post-processing should not print a Python traceback
    for expected missing or incomplete traces. Example: analyze.py ./empty
    returns 1 with a single "[error]" line.
    """
    args = parse_args()
    try:
        mode = fabricperf_mode()
        if mode == MEMORY_MODE:
            return analyze_memory_roots(args.trace_roots)
        if mode in THROUGHPUT_MODES:
            return analyze_throughput_roots(args.trace_roots, throughput_variant())
        return analyze_latency_roots(args.trace_roots, args.validate)
    except (FileNotFoundError, ImportError, KeyError, OSError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
