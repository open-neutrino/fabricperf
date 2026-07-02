"""Tests that inspect FabricPerf probe source files directly.

Functionality: assert important PTX snippets remain present or absent in the
TOML probe sources. Example: latency.probe must use root-relative offsets rather
than the old per-source direct offset table lookup.
"""

import unittest
from pathlib import Path

from neutrino.plugins.fabricperf.tests import _fabricperf_test_utils  # noqa: F401


class FabricPerfProbeSourceTests(unittest.TestCase):
    """Cover static source checks for latency and throughput probes."""

    def test_latency_probe_uses_generated_map_pointer(self):
        probe_path = Path(__file__).resolve().parents[1] / "latency.probe"
        probe_text = probe_path.read_text(encoding="utf-8")
        active_lines = [
            line
            for line in probe_text.splitlines()
            if not line.strip().startswith("//")
            and not line.strip().startswith("# OLD:")
        ]
        active_text = "\n".join(active_lines)

        self.assertIn("ld.global.u64 %ptp_rd76, [__neutrino_map_ptr_0];", probe_text)
        self.assertNotIn("ld.param.u64 %ptp_rd76, [param_ptp_metrics];", active_text)
        self.assertNotIn("setp.ne.s32 %ptp_p2, %ptp_r1, 2;", active_text)
        self.assertIn("setp.lt.u32 %ptp_p2, %ptp_r1, 2;", active_text)
        self.assertIn("setp.gt.u32 %ptp_p2, %ptp_r1, 8;", active_text)
        self.assertIn("@%ptp_p2 bra $L__ptp_done_exit;", active_text)
        self.assertNotIn("@%ptp_p2 bra $L__ptp_done;", active_text)
        self.assertIn("$L__ptp_leader_loop:", active_text)
        self.assertIn("root-clock calibration only runs leader rank 0", probe_text)
        self.assertIn("bra.uni $L__ptp_done;", active_text)
        self.assertNotIn("setp.lt.u32 %ptp_p20, %ptp_r30, %ptp_r1;", active_text)
        self.assertIn('capture.recv_step = { from = "nearby_before", op = "setp.lt.u64", operand = 2, window = 5 }', active_text)
        self.assertIn("capture.send_step = { operand = 1 }", active_text)
        self.assertIn("globalLatencyMailboxBuff = \"u64\"", active_text)
        self.assertIn("globalLatencyOffsetBuff = \"u64\"", active_text)
        self.assertIn("ptpRunId = \"u32\"", active_text)
        self.assertIn("ld.const.u64 %ptp_rd150, [globalLatencyOffsetBuff];", active_text)
        self.assertIn("st.global.u64 [%ptp_rd152], %ptp_rd147;", active_text)
        self.assertIn("st.global.u64 [%ptp_rd152], %ptp_rd98;", active_text)
        self.assertIn("ld.const.u64 %srlat_rd2, [globalLatencyMailboxBuff];", active_text)
        self.assertIn("ld.const.u64 %srmsg_rd1, [globalLatencyMailboxBuff];", active_text)
        self.assertIn("shl.b64 %srlat_rd11, ${recv_step}, 32;", active_text)
        self.assertIn("shl.b64 %srmsg_rd11, ${send_step}, 32;", active_text)
        self.assertIn("and.b32 %srlat_r5, %srlat_r5, 16777215;", active_text)
        self.assertIn("and.b32 %srmsg_r6, %srmsg_r6, 16777215;", active_text)
        self.assertNotIn("shl.b64 %srlat_rd11, ${recv_step}, 24;", active_text)
        self.assertNotIn("shl.b64 %srmsg_rd11, ${send_step}, 24;", active_text)
        self.assertIn("ld.const.u64 %srlat_rd14, [globalLatencyOffsetBuff];", active_text)
        self.assertIn("sender publishes root-clock timestamps", probe_text)
        self.assertIn("mov.u64 %srlat_rd23, %srlat_rd14;", active_text)
        self.assertNotIn("mul.wide.u32 %srlat_rd22, %srlat_r3, 8;", active_text)
        self.assertIn("ld.global.u64 %srlat_rd15, [%srlat_rd23];", active_text)
        self.assertIn("ld.const.u64 %srmsg_rd14, [globalLatencyOffsetBuff];", active_text)
        self.assertIn("sub.s64 %srmsg_rd5, %srmsg_rd5, %srmsg_rd15;", active_text)
        self.assertIn("SAVE [ sr_latency_debug ]", active_text)
        self.assertIn("[map.sr_latency_debug]", active_text)
        self.assertIn("setp.eq.u64 %srlat_p5, %srlat_rd13, 0;", active_text)
        self.assertIn("setp.le.u32 %srlat_p6, %srlat_r5, 1;", active_text)
        self.assertIn("or.pred %srlat_p5, %srlat_p5, %srlat_p6;", active_text)
        self.assertNotIn("mov.u64 %srlat_rd17, 9223372036854775807;", active_text)
        self.assertIn("add.u32 %srlat_r6, %srlat_r3, 1;", active_text)
        self.assertIn("shl.b32 %srlat_r6, %srlat_r6, 16;", active_text)
        self.assertIn("or.b32 %srlat_r6, %srlat_r6, %srlat_r4;", active_text)
        self.assertIn("@%srlat_p5 or.b32 %srlat_r6, %srlat_r6, 2147483648;", active_text)
        self.assertNotIn("setp.gt.s64 %srlat_p5, %srlat_rd17, 1000000000;", active_text)
        self.assertNotIn("ld.const.u64 %srlat_rd14, [globalResultBuff];", active_text)
        self.assertIn("cap = 1120", active_text)
        self.assertIn("capture.send_peer_mod", active_text)
        self.assertNotIn("setp.ne.u32 %srlat_p2, ${recv_peer}, %srlat_r3;", active_text)
        self.assertNotIn("setp.ne.u32 %srmsg_p2, ${send_peer}, %srmsg_r3;", active_text)
        self.assertIn("setp.lt.u32 %srlat_p4, %srlat_r5, 1048576;", active_text)
        self.assertIn("setp.lt.u32 %ptp_p22, %ptp_r6, 1048576;", active_text)

    def test_throughput_probe_has_no_atomics(self):
        probe = Path(__file__).parents[1] / "throughput.probe"
        text = probe.read_text(encoding="utf-8")
        self.assertNotIn("atom.global", text)
        self.assertNotIn("atom.shared", text)
        self.assertNotIn("fabricperf_throughput_hotpath", text)
        self.assertNotIn("fabricperfThroughputVariant", text)
