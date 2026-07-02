"""Tests for FabricPerf throughput analyzer output.

Functionality: validate throughput map decoding, CSV rows, and overflow summaries.
Example: one 4096-cell CTA stream splits into four workgroup slices.
"""

import csv
import tempfile
import unittest
from pathlib import Path

from neutrino.plugins.fabricperf.tests import _fabricperf_test_utils  # noqa: F401
from neutrino.plugins.fabricperf.analysis import throughput as throughput_analysis


class FabricPerfAnalyzerThroughputTests(unittest.TestCase):
    """Cover throughput-mode CSV rows and overflow summaries."""

    def test_throughput_records_write_csv_and_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trace_dir = root / "rank0"
            result_dir = trace_dir / "result"
            result_dir.mkdir(parents=True)
            (result_dir / "0.bin").write_bytes(b"")
            (trace_dir / "event.log").write_text(
                "[plugin][fabricperf] prepared rank=0 world=1 device=0\n",
                encoding="utf-8",
            )
            (trace_dir / "read.py").write_text(
                "\n".join([
                    "from collections import namedtuple",
                    "Thr = namedtuple('Thr', 'value')",
                    "def parse(path):",
                    "    slots = [Thr(0) for _ in range(4096)]",
                    "    slots[0] = Thr(20000)",
                    "    slots[1] = Thr(3)",
                    "    slots[2] = Thr(1000)",
                    "    slots[3] = Thr(9000)",
                    "    slots[4] = Thr(17000)",
                    "    slots[1024] = Thr(50000)",
                    "    slots[1025] = Thr(1024)",
                    "    for i in range(1022):",
                    "        slots[1026 + i] = Thr(i * 10)",
                    "    records = {",
                    "        'fabricperf_throughput': [[slots]],",
                    "    }",
                    "    return None, None, records",
                    "",
                ]),
                encoding="utf-8",
            )

            aggregates, csv_rows = throughput_analysis.read_throughput_root(root, "gmem")
            by_workgroup = {item.workgroup: item for item in aggregates}
            self.assertEqual(by_workgroup[0].captured, 3)
            self.assertEqual(by_workgroup[0].dropped, 0)
            self.assertEqual(by_workgroup[1].captured, 1022)
            self.assertEqual(by_workgroup[1].dropped, 2)

            throughput_analysis.write_throughput_csv(root, csv_rows)
            with (root / "fabricperf_throughput.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1025)
            self.assertEqual(rows[0]["variant"], "gmem")
            self.assertEqual(rows[0]["workgroup"], "0")
            self.assertEqual(rows[0]["bin_index"], "0")
            self.assertEqual(rows[-1]["workgroup"], "1")
            self.assertEqual(rows[-1]["sequence"], "1021")
            self.assertEqual(rows[-1]["bin_ns"], "8000")

    def test_throughput_records_decode_nonzero_warp_streams(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trace_dir = root / "rank0"
            result_dir = trace_dir / "result"
            result_dir.mkdir(parents=True)
            (result_dir / "0.bin").write_bytes(b"")
            (trace_dir / "event.log").write_text(
                "[plugin][fabricperf] prepared rank=0 world=1 device=0\n",
                encoding="utf-8",
            )
            (trace_dir / "read.py").write_text(
                "\n".join([
                    "from collections import namedtuple",
                    "Thr = namedtuple('Thr', 'value')",
                    "def parse(path):",
                    "    empty = [Thr(0) for _ in range(4096)]",
                    "    warp1 = [Thr(0) for _ in range(4096)]",
                    "    warp1[0] = Thr(10000)",
                    "    warp1[1] = Thr(2)",
                    "    warp1[2] = Thr(2000)",
                    "    warp1[3] = Thr(6000)",
                    "    records = {'fabricperf_throughput': [[empty, warp1]]}",
                    "    return None, None, records",
                    "",
                ]),
                encoding="utf-8",
            )

            aggregates, csv_rows = throughput_analysis.read_throughput_root(root, "gmem")

            self.assertEqual(len(csv_rows), 2)
            self.assertEqual(csv_rows[0]["workgroup"], "4")
            self.assertEqual(csv_rows[0]["sequence"], "0")
            self.assertEqual(csv_rows[1]["sequence"], "1")
            self.assertEqual(aggregates[0].workgroup, 4)
            self.assertEqual(aggregates[0].captured, 2)

    def test_throughput_records_decode_compact_cell_streams(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trace_dir = root / "rank0"
            result_dir = trace_dir / "result"
            result_dir.mkdir(parents=True)
            (result_dir / "0.bin").write_bytes(b"")
            (trace_dir / "event.log").write_text(
                "[plugin][fabricperf] prepared rank=0 world=1 device=0\n",
                encoding="utf-8",
            )
            (trace_dir / "read.py").write_text(
                "\n".join([
                    "from collections import namedtuple",
                    "Thr = namedtuple('Thr', 'value')",
                    "def parse(path):",
                    "    slots = [Thr(0) for _ in range(64)]",
                    "    slots[16] = Thr(42000)",
                    "    slots[17] = Thr(20)",
                    "    for i in range(14):",
                    "        slots[18 + i] = Thr(1000 + i)",
                    "    records = {'fabricperf_throughput': [[slots]]}",
                    "    return None, None, records",
                    "",
                ]),
                encoding="utf-8",
            )

            aggregates, csv_rows = throughput_analysis.read_throughput_root(root, "gmem")

            self.assertEqual(len(csv_rows), 14)
            self.assertEqual(csv_rows[0]["workgroup"], "1")
            self.assertEqual(csv_rows[-1]["sequence"], "13")
            by_workgroup = {item.workgroup: item for item in aggregates}
            self.assertEqual(by_workgroup[1].captured, 14)
            self.assertEqual(by_workgroup[1].dropped, 6)
