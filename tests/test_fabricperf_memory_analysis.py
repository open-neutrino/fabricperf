"""Tests for FabricPerf memory analyzer merge behavior.

Functionality: merge Neutrino probe byte counters into CUPTI PM rows. Example:
synthetic `fabricperf_probe_bytes` records fill DRAM and XBAR byte rates in the
merged `fabricperf_cupti.csv`.
"""

import csv
import tempfile
import unittest
from pathlib import Path

from neutrino.plugins.fabricperf.tests import _fabricperf_test_utils  # noqa: F401
from neutrino.plugins.fabricperf.analysis import memory as memory_analysis


class FabricPerfAnalyzerMemoryTests(unittest.TestCase):
    """Cover memory-mode merging of Neutrino probe bytes into CUPTI PM rows."""

    def test_memory_merge_combines_probe_bytes_and_pm_csv(self):
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
                    "Rec = namedtuple('Rec', 'ld_global_bytes st_global_bytes cp_async_bytes launch_index')",
                    "def parse(path):",
                    "    records = {'fabricperf_probe_bytes': [[[",
                    "        Rec(10, 5, 2, 0),",
                    "        Rec(6, 7, 1, 0),",
                    "    ]]]}",
                    "    return None, None, records",
                    "",
                ]),
                encoding="utf-8",
            )

            fields = [
                "rank",
                "device",
                "launch_index",
                "kernel",
                "grid",
                "block",
                "shared_bytes",
                "duration_s",
                "dram_read_Bps",
                "dram_write_Bps",
                "nvlink_rx_Bps",
                "nvlink_tx_Bps",
                "xbar_read_Bps",
                "xbar_write_Bps",
                "xbar_metric",
                "xbar_value",
                "raw_metrics",
            ]
            with (trace_dir / "fabricperf_cupti.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "rank": "0",
                    "device": "0",
                    "launch_index": "0",
                    "kernel": "kernel",
                    "grid": "1x1x1",
                    "block": "1x1x1",
                    "shared_bytes": "0",
                    "duration_s": "2",
                    "dram_read_Bps": "",
                    "dram_write_Bps": "",
                    "nvlink_rx_Bps": "50",
                    "nvlink_tx_Bps": "100",
                    "xbar_read_Bps": "",
                    "xbar_write_Bps": "",
                    "xbar_metric": "",
                    "xbar_value": "",
                    "raw_metrics": "cupti_backend=pm_sampling;nvlrx__bytes.sum=100;nvltx__bytes.sum=200;pm_samples_total=3",
                })

            self.assertTrue(memory_analysis.merge_one_pass_probe_output(root))

            with (root / "fabricperf_cupti.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["dram_read_Bps"], "9.5")
            self.assertEqual(row["dram_write_Bps"], "6")
            self.assertEqual(row["xbar_read_Bps"], "9.5")
            self.assertEqual(row["xbar_write_Bps"], "6")
            self.assertEqual(row["xbar_metric"], "fabricperf_probe_read_bytes")
            self.assertEqual(row["xbar_value"], "19")
            self.assertIn("probe_ld_global_bytes=16", row["raw_metrics"])
            self.assertIn("probe_cp_async_bytes=3", row["raw_metrics"])
            self.assertTrue((root / "fabricperf_cupti_metrics.csv").is_file())
            self.assertTrue((root / "fabricperf_cupti_diagnose.csv").is_file())

    def test_memory_merge_accepts_shared_root_pm_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trace_dir = root / "rank1"
            result_dir = trace_dir / "result"
            result_dir.mkdir(parents=True)
            (result_dir / "0.bin").write_bytes(b"")
            (trace_dir / "event.log").write_text(
                "[plugin][fabricperf] ready backend=pm_sampling rank=1 device=1 chip=GH100 csv=/tmp/root/fabricperf_cupti.csv\n",
                encoding="utf-8",
            )
            (trace_dir / "read.py").write_text(
                "\n".join([
                    "from collections import namedtuple",
                    "Rec = namedtuple('Rec', 'ld_global_bytes st_global_bytes cp_async_bytes launch_index')",
                    "def parse(path):",
                    "    records = {'fabricperf_probe_bytes': [[[Rec(8, 20, 4, 2)]]]}",
                    "    return None, None, records",
                    "",
                ]),
                encoding="utf-8",
            )

            fields = [
                "rank",
                "device",
                "launch_index",
                "kernel",
                "grid",
                "block",
                "shared_bytes",
                "duration_s",
                "dram_read_Bps",
                "dram_write_Bps",
                "nvlink_rx_Bps",
                "nvlink_tx_Bps",
                "xbar_read_Bps",
                "xbar_write_Bps",
                "xbar_metric",
                "xbar_value",
                "raw_metrics",
            ]
            with (root / "fabricperf_cupti.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "rank": "1",
                    "device": "1",
                    "launch_index": "2",
                    "kernel": "kernel",
                    "grid": "1x1x1",
                    "block": "1x1x1",
                    "shared_bytes": "0",
                    "duration_s": "4",
                    "dram_read_Bps": "",
                    "dram_write_Bps": "",
                    "nvlink_rx_Bps": "",
                    "nvlink_tx_Bps": "",
                    "xbar_read_Bps": "",
                    "xbar_write_Bps": "",
                    "xbar_metric": "",
                    "xbar_value": "",
                    "raw_metrics": "cupti_backend=pm_sampling;pm_samples_total=0",
                })

            self.assertTrue(memory_analysis.merge_one_pass_probe_output(root))

            with (root / "fabricperf_cupti.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["dram_read_Bps"], "3")
            self.assertEqual(rows[0]["dram_write_Bps"], "5")
            self.assertIn("probe_merge_key=launch_index", rows[0]["raw_metrics"])
