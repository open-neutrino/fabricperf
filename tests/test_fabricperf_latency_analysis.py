"""Tests for FabricPerf latency, PTP, and mailbox analysis helpers.

Functionality: cover latency CSV row generation, compact PTP diagnostic decoding,
and mailbox tag/meta bitfields. Example: root-relative offsets derive pairwise
`src -> dst` offsets by subtraction.
"""

import types
import struct
import tempfile
import unittest
from pathlib import Path
from typing import NamedTuple

from neutrino.plugins.fabricperf.tests import _fabricperf_test_utils  # noqa: F401
from neutrino.plugins.fabricperf.analysis import latency as latency_analysis
from neutrino.plugins.fabricperf.analysis import mailbox_debug
from neutrino.plugins.fabricperf.analysis import ptp_alignment


class FabricPerfAnalyzerLatencyTests(unittest.TestCase):
    """Cover latency analyzer handling for generated fixed-size maps."""

    def test_latency_analyzer_skips_unwritten_zero_slot(self):
        empty = types.SimpleNamespace(latency=0, vblock=0, step=0)
        sample = types.SimpleNamespace(latency=1200, vblock=0, step=0)

        self.assertTrue(latency_analysis.is_unwritten_latency_record(empty, "vblock"))
        self.assertFalse(latency_analysis.is_unwritten_latency_record(sample, "vblock"))

    def test_latency_analyzer_marks_invalid_sentinel(self):
        invalid = types.SimpleNamespace(
            latency=latency_analysis.LATENCY_INVALID_SENTINEL,
            vblock=3,
            step=42,
        )
        flagged = types.SimpleNamespace(
            latency=1200,
            vblock=latency_analysis.LATENCY_INVALID_FLAG | 3,
            step=42,
        )
        sample = types.SimpleNamespace(latency=1200, vblock=3, step=43)

        self.assertTrue(latency_analysis.is_invalid_latency_record(invalid))
        self.assertTrue(latency_analysis.is_invalid_latency_record(flagged))
        self.assertFalse(latency_analysis.is_invalid_latency_record(sample))

    def test_latency_analyzer_splits_packed_source_channel(self):
        packed = types.SimpleNamespace(latency=1200, vblock=((5 + 1) << 16) | 7, step=42)
        flagged = types.SimpleNamespace(
            latency=1200,
            vblock=latency_analysis.LATENCY_INVALID_FLAG | ((5 + 1) << 16) | 7,
            step=42,
        )
        source_zero = types.SimpleNamespace(latency=1200, vblock=(1 << 16) | 7, step=42)
        legacy = types.SimpleNamespace(latency=1200, vblock=7, step=42)

        self.assertEqual(latency_analysis.split_latency_channel(packed, "vblock"), (5, 7))
        self.assertEqual(latency_analysis.split_latency_channel(flagged, "vblock"), (5, 7))
        self.assertEqual(latency_analysis.split_latency_channel(source_zero, "vblock"), (0, 7))
        self.assertEqual(
            latency_analysis.split_latency_channel(legacy, "vblock"),
            (latency_analysis.LATENCY_SOURCE_UNKNOWN, 7),
        )

    def test_latency_rows_exclude_invalid_sentinel_from_stats(self):
        stats = latency_analysis.TraceStats("trace", 0, "vblock")
        stats.channels[(2, 3)].add(1000, 10)
        stats.channels[(2, 3)].add(3000, 12)
        stats.channels[(2, 3)].add_invalid(11)

        rows = latency_analysis.rows_for_stats(stats)

        self.assertEqual(rows[0][2], "2")
        self.assertEqual(rows[0][4], "3")
        self.assertEqual(rows[0][5], "2")
        self.assertEqual(rows[0][6], "1")
        self.assertEqual(rows[0][7], "33.33")
        self.assertEqual(rows[0][8], "2")
        self.assertEqual(rows[0][9], "1")
        self.assertEqual(rows[0][10], "2")
        self.assertEqual(rows[0][11], "3")
        self.assertEqual(rows[0][12], "10..12")


class FabricPerfPtpAlignmentTests(unittest.TestCase):
    """Cover PTP offset closure helpers for compact diagnostic rows."""

    def test_ptp_records_decode_deepep_two_section_layout(self):
        class PtpMetrics(NamedTuple):
            """Synthetic generated `ptp_metrics` row."""

            offset: int
            latency: int

        with tempfile.TemporaryDirectory() as tmp:
            result_file = f"{tmp}/result.bin"
            sr_section_size = 64
            ptp_section_size = 17920
            grid_size = 2
            warp_streams = 2
            ptp_offset = 64 + (grid_size * warp_streams * sr_section_size)
            header = struct.pack("iiiiiiii", grid_size, 1, 1, 64, 1, 1, 0, 2)
            sr_section = struct.pack("IIQ", sr_section_size, 32, 64)
            ptp_section = struct.pack("IIQ", ptp_section_size, 32, ptp_offset)
            ptp_payload = bytearray(ptp_section_size)
            slot_index = (0 * 7 + 0) * 20 + 3
            struct.pack_into("qq", ptp_payload, slot_index * 16, 123, 45)
            with open(result_file, "wb") as handle:
                handle.write(header)
                handle.write(sr_section)
                handle.write(ptp_section)
                handle.write(b"\0" * (ptp_offset - handle.tell()))
                handle.write(ptp_payload)

            rows = ptp_alignment.parse_ptp_records(
                types.SimpleNamespace(ptp_metrics=PtpMetrics),
                Path(result_file),
            )

        self.assertEqual(rows[slot_index], PtpMetrics(123, 45))

    def test_ptp_slot_decode_expands_compact_destination(self):
        self.assertEqual(ptp_alignment.decode_ptp_slot((0 * 7 + 0) * 20 + 3), (0, 1, 3))
        self.assertEqual(ptp_alignment.decode_ptp_slot((1 * 7 + 0) * 20 + 4), (1, 0, 4))
        self.assertEqual(ptp_alignment.decode_ptp_slot((1 * 7 + 2) * 20 + 5), (1, 3, 5))

    def test_ptp_closure_residuals_are_signed(self):
        offsets = {
            (0, 1): [10, 11, 12],
            (1, 2): [20, 21, 22],
            (0, 2): [31, 35, 33],
        }

        residuals = ptp_alignment.closure_residuals(offsets, 0, 1, 2)

        self.assertEqual(residuals, [1, 3, -1])

    def test_ptp_reciprocal_residuals_are_signed(self):
        offsets = {
            (0, 1): [10, 11, 12],
            (1, 0): [-9, -12, -10],
        }

        residuals = ptp_alignment.reciprocal_residuals(offsets, 0, 1)

        self.assertEqual(residuals, [1, -1, 2])

    def test_ptp_root_relative_offsets_derive_pairs_by_subtraction(self):
        offsets = {
            (0, 1): [10, 11],
            (0, 2): [30, 32],
        }

        derived = ptp_alignment.derive_root_relative_offsets(offsets, devices=3)

        self.assertEqual(derived[(0, 1)], [10, 11])
        self.assertEqual(derived[(1, 0)], [-10, -11])
        self.assertEqual(derived[(1, 2)], [20, 21])
        self.assertEqual(derived[(2, 1)], [-20, -21])


class FabricPerfMailboxDebugTests(unittest.TestCase):
    """Cover compact mailbox-debug bitfields."""

    def test_decode_meta_unpacks_poll_source_and_vblock(self):
        meta = (17 << 32) | ((3 + 1) << 16) | 9

        self.assertEqual(mailbox_debug.decode_meta(meta), (17, 3, 9))

    def test_decode_meta_ignores_invalid_flag(self):
        meta = (17 << 32) | mailbox_debug.LATENCY_INVALID_FLAG | ((3 + 1) << 16) | 9

        self.assertEqual(mailbox_debug.decode_meta(meta), (17, 3, 9))
        self.assertTrue(mailbox_debug.decode_invalid(meta, 1200))

    def test_decode_tag_uses_step_high_32_and_masked_run_id(self):
        tag = (123 << 32) | (0xABCDEF << 8) | mailbox_debug.TAG_KIND

        self.assertEqual(mailbox_debug.decode_tag(tag), (123, 0xABCDEF, mailbox_debug.TAG_KIND))
