"""FabricPerf DeepEP prober specialization tests."""

import os
import unittest
from unittest import mock
import sys
import types

sys.modules.setdefault("toml", types.SimpleNamespace())

from neutrino.plugins.fabricperf.deepep import prober


class FabricPerfDeepEpProberTests(unittest.TestCase):
    """Cover DeepEP-specific probe rewrites.

    Motivation: DeepEP V2 communication kernels expose TMA wait anchors instead
    of NCCL devFunc polling. Example: latency runtime exchange records mailbox
    timing around `cp.async.bulk.wait_group 0`.
    """

    def test_identifies_deepep_ep_kernels(self):
        self.assertTrue(prober.is_deepep_kernel("_ZN7deep_ep7elastic13dispatch_implFoo"))
        self.assertTrue(prober.is_deepep_kernel("_ZN7deep_ep7elastic12combine_implFoo"))
        self.assertFalse(prober.is_deepep_kernel("_ZN7deep_ep7elastic7barrierFoo"))

    def test_specializes_throughput_to_tma_wait(self):
        raw_probe = {
            "regs": 3,
            "probe": {
                "fabricperf_throughput_recv": {
                    "level": "thread",
                    "match": {"kind": "branch"},
                    "capture": {"recv_step": {"operand": 2}},
                    "after": "setp.lt.u64 %p, %NR1, 1022;\n"
                    "mul.wide.u32 %rd, %r2, 4096;",
                },
                "fabricperf_throughput_flush": {
                    "level": "thread",
                    "before": "mul.wide.u32 %rd, %r2, 4096;",
                },
            },
            "map": {
                "fabricperf_throughput": {
                    "level": "warp",
                    "type": "array",
                    "size": 4,
                    "cap": 4096,
                    "regs": {"value": ["u32", "None"]},
                }
            },
        }

        specialized = prober.specialize_throughput_probe(raw_probe)

        recv = specialized["probe"]["fabricperf_throughput_recv"]
        self.assertEqual(recv["match"], {"kind": "deepep_tma_wait", "last": True})
        self.assertFalse(recv["inherit_predicate"])
        self.assertEqual(recv["capture"]["group_source"]["value"], "%tid.x")
        self.assertEqual(recv["capture"]["vblock"]["value"], "%ctaid.x")
        self.assertIn("2", recv["after"])
        self.assertIn("16", recv["after"])
        self.assertIn("16", specialized["probe"]["fabricperf_throughput_flush"]["before"])
        self.assertEqual(specialized["map"]["fabricperf_throughput"]["cap"], 16)
        self.assertEqual(raw_probe["probe"]["fabricperf_throughput_recv"]["match"], {"kind": "branch"})

    def test_specializes_latency_to_runtime_exchange_by_default(self):
        raw_probe = {
            "regs": 0,
            "probe": {
                "recv_latency": {
                    "level": "thread",
                    "match": {"kind": "branch"},
                    "capture": {"recv_step": {"operand": 1}},
                    "after": "\n".join([
                        "{",
                        "    .reg .pred %srlat_p<7>;",
                        "    .reg .b32 %srlat_r<7>;",
                        "    .reg .b64 %srlat_rd<28>;",
                        "    ld.const.u64 %srlat_rd2, [globalLatencyMailboxBuff];",
                        "    SAVE [ sr_latency_debug ] { %srlat_rd17, %srlat_rd24 };",
                        "$L__srlat_done:",
                        "}",
                    ]),
                },
                "send_timestamp_message": {
                    "level": "thread",
                    "match": {"op": "st.volatile.global.u64"},
                    "capture": {"send_step": {"operand": 1}},
                    "after": "\n".join([
                        "{",
                        "    .reg .pred %srmsg_p<5>;",
                        "    .reg .b32 %srmsg_r<8>;",
                        "    .reg .b64 %srmsg_rd<18>;",
                        "    ld.const.u64 %srmsg_rd1, [globalLatencyMailboxBuff];",
                        "$L__srmsg_done:",
                        "}",
                    ]),
                },
                "ptp_metrics": {
                    "level": "thread",
                    "pos": "kernel",
                    "before": "\n".join([
                        "ld.const.u64 %ptp_rd1, [globalLeaderBuff];",
                        "ld.global.u64 %ptp_rd76, [__neutrino_map_ptr_0];",
                    ]),
                },
            },
            "symbol": {
                "globalLatencyMailboxBuff": "u64",
                "globalLatencyOffsetBuff": "u64",
                "globalLeaderBuff": "u64",
            },
            "map": {
                "ptp_metrics": {
                    "level": "warp",
                    "type": "array",
                    "size": 16,
                    "cap": 1,
                    "regs": {
                        "offset": ["u64", "None"],
                        "latency": ["u64", "None"],
                    },
                },
                "sr_latency": {
                    "level": "thread",
                    "type": "array",
                    "size": 24,
                    "cap": 1,
                    "regs": {
                        "latency": ["u64", "None"],
                        "source": ["u64", "None"],
                        "step": ["u64", "None"],
                    },
                },
                "sr_latency_debug": {
                    "level": "thread",
                    "type": "array",
                    "size": 64,
                    "cap": 1,
                    "regs": {
                        "latency": ["u64", "None"],
                        "meta": ["u64", "None"],
                    },
                },
            },
        }

        with mock.patch.dict(os.environ, {}, clear=True):
            specialized = prober.specialize_latency_probe(raw_probe)

        recv = specialized["probe"]["recv_latency"]
        send = specialized["probe"]["send_timestamp_message"]
        self.assertEqual(recv["match"], {"kind": "deepep_tma_wait", "last": True})
        self.assertEqual(send["match"], {"kind": "deepep_tma_wait", "last": True})
        self.assertIn("before", send)
        self.assertNotIn("after", send)
        self.assertEqual(recv["capture"]["recv_peer"]["value"], "%srlat_r3")
        self.assertEqual(send["capture"]["send_peer"]["value"], "%srmsg_r3")
        self.assertEqual(send["capture"]["send_peer_mod"]["value"], "4294967295")
        self.assertIn("one mailbox reader per CTA", recv["after"])
        self.assertIn("one mailbox writer per CTA", send["before"])
        self.assertNotIn("ptp_metrics", specialized["probe"])
        self.assertNotIn("globalLeaderBuff", specialized["symbol"])
        self.assertIn("globalLatencyMailboxBuff", specialized["symbol"])
        self.assertEqual(specialized["map"]["sr_latency"]["cap"], 4)
        self.assertNotIn("sr_latency_debug", specialized["map"])
        self.assertNotIn("SAVE [ sr_latency_debug ]", recv["after"])

        with mock.patch.dict(os.environ, {prober.DEEPEP_LATENCY_MODE_ENV: prober.DEEPEP_LATENCY_PTP_MODE}, clear=True):
            ptp_specialized = prober.specialize_latency_probe(raw_probe)

        self.assertIn("ptp_metrics", ptp_specialized["probe"])
        self.assertIn("ptp_metrics", ptp_specialized["map"])
        self.assertIn("globalLeaderBuff", ptp_specialized["symbol"])
        self.assertIn("globalFollowerBuff", ptp_specialized["symbol"])
        self.assertIn("globalResultBuff", ptp_specialized["symbol"])
        self.assertIn("ptpGlobalBarrier", ptp_specialized["symbol"])
        self.assertIn("ptpGlobalBarrierSense", ptp_specialized["symbol"])
        self.assertEqual(ptp_specialized["probe"]["recv_latency"]["match"], {"kind": "deepep_tma_wait", "last": True})
        self.assertIn("one mailbox reader per CTA", ptp_specialized["probe"]["recv_latency"]["after"])
        self.assertEqual(list(ptp_specialized["map"].keys()), ["sr_latency", "ptp_metrics"])
        self.assertIn("__neutrino_map_ptr_1", ptp_specialized["probe"]["ptp_metrics"]["before"])
        self.assertNotIn("__neutrino_map_ptr_0];", ptp_specialized["probe"]["ptp_metrics"]["before"])
        self.assertEqual(ptp_specialized["map"]["sr_latency"]["cap"], 4)

    def test_specializes_latency_to_local_tma_wait_window(self):
        raw_probe = {
            "regs": 0,
            "probe": {"recv_latency": {"level": "thread", "match": {"kind": "branch"}}},
            "map": {
                "sr_latency": {
                    "level": "thread",
                    "type": "array",
                    "size": 24,
                    "cap": 1,
                    "regs": {
                        "latency": ["u64", "None"],
                        "source": ["u64", "None"],
                        "step": ["u64", "None"],
                    },
                }
            },
        }

        specialized = prober.specialize_latency_local_probe(raw_probe)

        recv = specialized["probe"]["recv_latency"]
        self.assertEqual(recv["match"], {"kind": "deepep_tma_wait", "last": True})
        self.assertIn("mov.u64 %NR0, %globaltimer", recv["before"])
        self.assertIn("SAVE [ sr_latency ]", recv["after"])
        self.assertEqual(specialized["symbol"], {"deviceId": "u32"})


if __name__ == "__main__":
    unittest.main()
