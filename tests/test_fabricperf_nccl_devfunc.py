"""Tests for FabricPerf selected NCCL device-function probing.

Functionality: cover NCCL runtime table matching, generated declaration hoisting,
and latency/throughput fallback probe rewrites. Example: SendRecv device kernels
should redirect selected function dispatch to a local fallback path.
"""

import tempfile
import unittest
from pathlib import Path

from neutrino.plugins.fabricperf.tests import _fabricperf_test_utils  # noqa: F401
from neutrino.plugins.fabricperf.nccl import prober as nccl_devfunc_prober
from neutrino.plugins.fabricperf.nccl import table as nccl_devfunc_table


class FabricPerfNcclDevfuncTests(unittest.TestCase):
    """Cover NCCL devfunc target selection and probe rewrite helpers."""

    def test_nccl_devfunc_table_selects_sendrecv(self):
        ptx = """
.visible .func _Z20ncclDevFunc_SendRecvv()
{
    ret;
}
"""
        entry = """
.visible .entry ncclDevKernel_SendRecv()
{
    setp.eq.s16 %p1, %rs1, 669;
    ret;
}
"""

        target = nccl_devfunc_table.select_target("ncclDevKernel_SendRecv", ptx, entry, {})

        self.assertIsNotNone(target)
        self.assertEqual(target.function_name, "_Z20ncclDevFunc_SendRecvv")
        self.assertEqual(target.function_id, 669)

    def test_nccl_devfunc_table_selects_collective_descriptor(self):
        ptx = """
.visible .func _Z29ncclDevFunc_Broadcast_RING_LLv()
{
    ret;
}
.visible .func _Z32ncclDevFunc_Broadcast_RING_LL128v()
{
    ret;
}
.visible .func _Z33ncclDevFunc_Broadcast_RING_SIMPLEv()
{
    ret;
}
"""
        entry = """
.visible .entry _Z31ncclDevKernel_Broadcast_RING_LL24ncclDevKernelArgsStorageILm4096EE()
{
    setp.eq.s16 %p1, %rs1, 342;
    ret;
}
"""

        target = nccl_devfunc_table.select_target(
            "_Z31ncclDevKernel_Broadcast_RING_LL24ncclDevKernelArgsStorageILm4096EE",
            ptx,
            entry,
            {},
        )

        self.assertIsNotNone(target)
        self.assertEqual(target.function_name, "_Z29ncclDevFunc_Broadcast_RING_LLv")
        self.assertEqual(target.function_id, 342)
        self.assertEqual(target.source, "kernel-descriptor-match")

    def test_nccl_devfunc_prober_hoists_generated_decls(self):
        ptx = """.version 8.7
.visible .func _Z20ncclDevFunc_SendRecvv()
{
    ld.global.u64 %rd1, [__neutrino_map_ptr_0];
    ld.const.u32 %r1, [launchIndex];
    ret;
}
.visible .global .align 8 .u64 __neutrino_map_ptr_0;
.const .align 4 .u32 launchIndex;
.visible .entry ncclDevKernel_SendRecv()
{
    ret;
}
"""
        raw_probe = {"symbol": {"launchIndex": "u32"}}

        rewritten = nccl_devfunc_prober.hoist_generated_decls(ptx, raw_probe)

        self.assertLess(
            rewritten.index("__neutrino_map_ptr_0;"),
            rewritten.index(".visible .func _Z20ncclDevFunc_SendRecvv"),
        )
        self.assertLess(
            rewritten.index("launchIndex;"),
            rewritten.index(".visible .func _Z20ncclDevFunc_SendRecvv"),
        )
        self.assertGreater(
            rewritten.index("ld.const.u32 %r1, [launchIndex];"),
            rewritten.index(".visible .func _Z20ncclDevFunc_SendRecvv"),
        )
        self.assertEqual(rewritten.count("__neutrino_map_ptr_0;"), 1)
        self.assertEqual(rewritten.count("launchIndex;"), 1)

    def test_nccl_devfunc_latency_shell_keeps_ptp_at_entry(self):
        raw_probe = {
            "regs": 0,
            "map": {
                "ptp_metrics": {},
                "sr_latency": {},
            },
            "symbol": {
                "globalResultBuff": "u64",
            },
            "probe": {
                "ptp_metrics": {
                    "level": "thread",
                    "pos": "kernel",
                    "before": "\n".join([
                        "    ld.param.u64 %ptp_rd76, [param_ptp_metrics];",
                        "    cvta.to.global.u64 %ptp_rd76, %ptp_rd76;",
                        "    SAVE [ ptp_metrics ] { %rd1, %rd2 };",
                    ]),
                },
                "recv_latency": {
                    "level": "thread",
                    "match": {"kind": "branch"},
                    "after": "SAVE [ sr_latency ] { %rd3, %rd4, %rd5 };",
                },
            },
        }

        shell = nccl_devfunc_prober.shell_probe_for(raw_probe)

        self.assertEqual(list(shell["probe"].keys()), ["ptp_metrics"])
        self.assertIsNot(shell["probe"]["ptp_metrics"], raw_probe["probe"]["ptp_metrics"])
        self.assertIn("ld.global.u64 %ptp_rd76, [__neutrino_map_ptr_0];", shell["probe"]["ptp_metrics"]["before"])
        self.assertNotIn("param_ptp_metrics", shell["probe"]["ptp_metrics"]["before"])
        self.assertEqual(list(shell["map"].keys()), ["ptp_metrics", "sr_latency"])
        self.assertEqual(shell["symbol"], {"globalResultBuff": "u64"})

    def test_nccl_devfunc_non_latency_shell_is_noop(self):
        raw_probe = {
            "map": {"fabricperf_throughput": {}},
            "probe": {
                "fabricperf_throughput_recv": {
                    "level": "thread",
                    "match": {"op": "ld.global.u32"},
                    "after": "SAVE [ fabricperf_throughput ] { %rd1 };",
                }
            },
        }

        shell = nccl_devfunc_prober.shell_probe_for(raw_probe)

        self.assertEqual(list(shell["probe"].keys()), ["fabricperf_nccl_devfunc_shell"])
        self.assertEqual(shell["map"], raw_probe["map"])

    def test_nccl_devfunc_selected_latency_uses_local_probe(self):
        raw_probe = {
            "regs": 0,
            "map": {
                "ptp_metrics": {},
                "sr_latency": {
                    "regs": {
                        "latency": ["u64", "None"],
                        "vblock": ["u32", "None"],
                        "step": ["u32", "None"],
                    }
                },
                "sr_latency_debug": {},
            },
            "symbol": {
                "globalLatencyMailboxBuff": "u64",
                "deviceId": "u32",
            },
            "probe": {
                "ptp_metrics": {"before": "SAVE [ ptp_metrics ] { %rd1, %rd2 };"},
                "recv_latency": {"after": "SAVE [ sr_latency ] { %rd3, %rd4, %rd5 };"},
                "send_timestamp_message": {"after": "st.global.u64 [%rd1], %rd2;"},
            },
        }

        active, label = nccl_devfunc_prober.selected_devfunc_probe_config(raw_probe)

        self.assertEqual(label, "latency-local-devfunc")
        self.assertEqual(list(active["map"].keys()), ["sr_latency"])
        self.assertEqual(active["symbol"], {"deviceId": "u32"})
        self.assertIn("fabricperf_latency_local_epoch", active["probe"])
        self.assertNotIn("recv_latency_simple_b", active["probe"])
        self.assertNotIn("globalLatencyMailboxBuff", active.get("symbol", {}))
        shell = nccl_devfunc_prober.shell_probe_for(active)
        self.assertEqual(list(shell["probe"].keys()), ["fabricperf_nccl_devfunc_shell"])

    def test_nccl_devfunc_throughput_fallback_relaxes_entry_signature(self):
        raw_probe = {
            "map": {
                "fabricperf_throughput": {},
            },
            "probe": {
                "fabricperf_throughput_recv": {
                    "match": {
                        "kind": "branch",
                        "predicated": True,
                        "branch_target_is_previous_label": True,
                        "nearby_before_ops": ["ld.volatile.global.u64", "setp.lt.u64"],
                        "nearby_window": 5,
                        "last": True,
                    },
                    "after": "\n".join([
                        "    mov.u64 %fpthr_recv_rd5, %map_fabricperf_throughput1;",
                        "$L__fpthr_recv_done:",
                    ]),
                },
                "fabricperf_throughput_flush": {
                    "before": "\n".join([
                        "    .reg .b32 %fpthr_flush_r<4>;",
                        "    // Step: cell 0 is duration; cell 1 is total arrivals.",
                        "    mov.u64 %fpthr_flush_rd5, %map_fabricperf_throughput1;",
                    ]),
                }
            }
        }

        fallback = nccl_devfunc_prober.throughput_devfunc_fallback_probe(raw_probe)

        self.assertIsNotNone(fallback)
        match = fallback["probe"]["fabricperf_throughput_recv"]["match"]
        self.assertEqual(match["op"], "ld.volatile.global.v2.b64")
        self.assertEqual(match["nth"], 10)
        second_match = fallback["probe"]["fabricperf_throughput_recv_simple_b"]["match"]
        self.assertEqual(second_match["op"], "ld.volatile.global.v2.b64")
        self.assertEqual(second_match["nth"], 26)
        capture = fallback["probe"]["fabricperf_throughput_recv"]["capture"]
        self.assertEqual(capture["group"]["value"], "0")
        self.assertEqual(capture["group_source"]["value"], "%laneid")
        self.assertEqual(capture["group_threads"]["value"], "1")
        self.assertEqual(capture["recv_step"]["value"], "0")
        self.assertEqual(capture["recv_peer"]["value"], "0")
        self.assertEqual(capture["vblock"]["value"], "%ctaid.x")
        self.assertEqual(capture["chunk"]["value"], "0")
        after = fallback["probe"]["fabricperf_throughput_recv"]["after"]
        self.assertIn("mov.u64 %fpthr_recv_rd5, %map_fabricperf_throughput1;", after)
        second_after = fallback["probe"]["fabricperf_throughput_recv_simple_b"]["after"]
        self.assertIn("mov.u64 %fpthr_recv_b_rd5, %map_fabricperf_throughput1;", second_after)
        self.assertIn("$L__fpthr_recv_b_done:", second_after)
        flush = fallback["probe"]["fabricperf_throughput_flush"]["before"]
        self.assertIn(".reg .b32 %fpthr_flush_r<4>;", flush)
        self.assertIn("mov.u64 %fpthr_flush_rd5, %map_fabricperf_throughput1;", flush)
        self.assertEqual(
            raw_probe["probe"]["fabricperf_throughput_recv"]["match"]["nearby_before_ops"],
            ["ld.volatile.global.u64", "setp.lt.u64"],
        )
        self.assertNotIn("capture", raw_probe["probe"]["fabricperf_throughput_recv"])
        self.assertNotIn("fabricperf_throughput_recv_simple_b", raw_probe["probe"])
        self.assertIn(".reg .b32 %fpthr_flush_r<4>;", raw_probe["probe"]["fabricperf_throughput_flush"]["before"])

        latency = nccl_devfunc_prober.latency_devfunc_fallback_probe({
            "probe": {
                "recv_latency": {
                    "match": {
                        "kind": "branch",
                        "branch_target_is_previous_label": True,
                    },
                    "after": "\n".join([
                        "    .reg .b64 %srlat_rd<22>;",
                        "    add.s64 %srlat_rd9, %srlat_rd5, %srlat_rd8;",
                        "$L__srlat_wait_msg:",
                        "    ld.global.volatile.v2.u64 { %srlat_rd11, %srlat_rd12 }, [%srlat_rd9];",
                        "    setp.ne.s64 %srlat_p3, %srlat_rd11, %srlat_rd10;",
                        "    @%srlat_p3 bra $L__srlat_wait_msg;",
                        "$L__srlat_done:",
                    ]),
                },
                "send_timestamp_message": {
                    "match": {
                        "op": "st.volatile.global.u64",
                        "before_ref": "recv_latency",
                        "last": True,
                    },
                    "after": "\n".join([
                        "    .reg .b64 %srmsg_rd<13>;",
                        "$L__srmsg_done:",
                    ]),
                },
                "ptp_metrics": {
                    "level": "thread",
                    "pos": "kernel",
                    "before": "SAVE [ ptp_metrics ] { %rd1, %rd2 };",
                },
            }
        })
        self.assertNotIn("ptp_metrics", latency["probe"])
        self.assertEqual(latency["probe"]["recv_latency"]["match"]["op"], "ld.volatile.global.v2.b64")
        self.assertEqual(latency["probe"]["recv_latency"]["match"]["nth"], 10)
        self.assertEqual(latency["probe"]["recv_latency"]["capture"]["recv_step"]["value"], "0")
        self.assertEqual(latency["probe"]["recv_latency"]["capture"]["recv_peer"]["value"], "%srlat_r3")
        self.assertEqual(latency["probe"]["recv_latency"]["capture"]["vblock"]["from"], "virtual_block")
        self.assertIn("selected-devFunc latency uses one warp-lane writer", latency["probe"]["recv_latency"]["after"])
        self.assertIn("@%srlat_p4 bra $L__srlat_done;", latency["probe"]["recv_latency"]["after"])
        self.assertNotIn("publish this receive-arrival timestamp", latency["probe"]["recv_latency"]["after"])
        self.assertIn("avoid deadlocking NCCL", latency["probe"]["recv_latency"]["after"])
        self.assertIn("setp.lt.u32 %srlat_p3, %srlat_r4, 1024;", latency["probe"]["recv_latency"]["after"])
        self.assertIn("bra.uni $L__srlat_done;", latency["probe"]["recv_latency"]["after"])
        self.assertIn("$L__srlat_have_msg:", latency["probe"]["recv_latency"]["after"])
        self.assertEqual(latency["probe"]["send_timestamp_message"]["match"]["op"], "ld.volatile.global.u64")
        self.assertEqual(latency["probe"]["send_timestamp_message"]["match"]["before_ref"], "recv_latency")
        self.assertTrue(latency["probe"]["send_timestamp_message"]["match"]["last"])
        self.assertEqual(latency["probe"]["send_timestamp_message"]["capture"]["send_step"]["value"], "0")
        self.assertEqual(latency["probe"]["send_timestamp_message"]["capture"]["send_peer"]["value"], "%srmsg_r3")
        self.assertIn("@%srmsg_p3 bra $L__srmsg_done;", latency["probe"]["send_timestamp_message"]["after"])

    def test_nccl_devfunc_rewrites_legacy_latency_param_load(self):
        body = """
    ld.param.u64 %ptp_rd76, [param_ptp_metrics];
    cvta.to.global.u64 %ptp_rd76, %ptp_rd76;
"""
        raw_probe = {"map": {"ptp_metrics": {}, "sr_latency": {}}}

        rewritten = nccl_devfunc_prober.rewrite_legacy_latency_param_load(body, raw_probe)

        self.assertIn("ld.global.u64 %ptp_rd76, [__neutrino_map_ptr_0];", rewritten)
        self.assertNotIn("param_ptp_metrics", rewritten)

    def test_nccl_devfunc_accepts_already_bounded_latency_wait(self):
        snippet = "\n".join([
            "$L__srlat_wait_msg:",
            "    mov.u32 %srlat_r4, 0;",
            "$L__srlat_wait_msg_loop:",
            "    ld.global.volatile.v2.u64 { %srlat_rd11, %srlat_rd12 }, [%srlat_rd9];",
            "    setp.eq.s64 %srlat_p3, %srlat_rd11, %srlat_rd10;",
            "    @%srlat_p3 bra $L__srlat_have_msg;",
            "    add.u32 %srlat_r4, %srlat_r4, 1;",
            "    setp.lt.u32 %srlat_p3, %srlat_r4, 1048576;",
            "    @%srlat_p3 bra $L__srlat_wait_msg_loop;",
            "    bra.uni $L__srlat_done;",
            "$L__srlat_have_msg:",
        ])

        self.assertIn("setp.lt.u32 %srlat_p3, %srlat_r4, 1024;", nccl_devfunc_prober.bound_latency_devfunc_recv_wait(snippet))

    def test_nccl_devfunc_publishes_new_latency_mailbox(self):
        snippet = "\n".join([
            "    .reg .b64 %srlat_rd<28>;",
            "    add.s64 %srlat_rd10, %srlat_rd5, %srlat_rd9;",
            "    shl.b64 %srlat_rd11, ${recv_step}, 32;",
            "    or.b64 %srlat_rd11, %srlat_rd11, 160;",
            "$L__srlat_wait_msg:",
        ])

        rewritten = nccl_devfunc_prober.inject_latency_devfunc_recv_publish(snippet)

        self.assertIn(".reg .b64 %srlat_rd<28>;", rewritten)
        self.assertIn("publish this receive-arrival timestamp to the peer mailbox", rewritten)
        self.assertGreater(rewritten.index("or.b64 %srlat_rd11"), rewritten.index("shl.b64 %srlat_rd11"))
        self.assertGreater(rewritten.index("st.global.volatile.u64 [%srlat_rd26], %srlat_rd11;"), rewritten.index("or.b64 %srlat_rd11"))
        self.assertIn("st.global.volatile.u64 [%srlat_rd27], %srlat_rd1;", rewritten)

    def test_nccl_devfunc_redirects_direct_dispatch_to_fallback(self):
        ptx = """
    setp.eq.s16 %p39, %rs4, 669;
    // discovery block can sit between compare and branch.
    mov.u32 %r1, %r1;
    @%p39 bra $L__BB1_47;
    bra.uni $L__BB1_46;
$L__BB1_46:
    call.uni %rd1, ();
$L__BB1_47:
    ret;
"""

        rewritten, redirects = nccl_devfunc_prober.redirect_direct_dispatch_to_fallback(ptx, 669)

        self.assertEqual(redirects, 1)
        self.assertIn("// OLD: @%p39 bra $L__BB1_47;", rewritten)
        self.assertIn("@%p39 bra $L__BB1_46;", rewritten)
        self.assertIn("bra.uni $L__BB1_46;", rewritten)

    def test_nccl_devfunc_redirects_setp_ne_direct_dispatch_to_fallback(self):
        ptx = """
    setp.ne.s16 %p47, %rs7, 669;
    @%p47 bra $L__BB1_609;
    mov.u32 %r1, %r1;
$L__BB1_609:
    call.uni %rd1, ();
"""

        rewritten, redirects = nccl_devfunc_prober.redirect_direct_dispatch_to_fallback(ptx, 669)

        self.assertEqual(redirects, 1)
        self.assertIn("// OLD: @%p47 bra $L__BB1_609;", rewritten)
        self.assertIn("bra.uni $L__BB1_609;", rewritten)
        self.assertNotIn("\n    @%p47 bra $L__BB1_609;", rewritten)

    def test_nccl_devfunc_infers_func_id_from_runtime_table_mapping(self):
        target = nccl_devfunc_table.NcclDevFuncTarget(
            kernel_name="ncclDevKernel_SendRecv",
            function_name="_Z20ncclDevFunc_SendRecvv",
            function_id=None,
            source="single-local-devfunc",
            candidates=("_Z20ncclDevFunc_SendRecvv",),
        )

        self.assertEqual(
            nccl_devfunc_prober.selected_target_func_id(
                target,
                {669: "_Z20ncclDevFunc_SendRecvv"},
            ),
            669,
        )

    def test_nccl_devfunc_parses_runtime_table_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kernel_info = Path(tmpdir) / "kernel.info"
            kernel_info.write_text(
                "\n".join([
                    "_Z31ncclDevKernel_AllGather_RING_LL24ncclDevKernelArgsStorageILm4096EE",
                    "1",
                    "1",
                    "1",
                    "0,32",
                    "launchIndex,0,1",
                    "NCCL_RUNTIME_TABLE",
                    "__neutrino_patch_ncclDevFuncTable",
                    "ncclDevFuncTable",
                    "__neutrino_nccl_discovery_mode",
                    "__neutrino_nccl_runtime_func_id",
                    "671",
                    "3",
                    "3,_Z29ncclDevFunc_AllGather_RING_LLv",
                    "4,_Z32ncclDevFunc_AllGather_RING_LL128v",
                    "5,_Z33ncclDevFunc_AllGather_RING_SIMPLEv",
                ]),
                encoding="utf-8",
            )

            mapping = nccl_devfunc_prober.nccl_runtime_table_mapping_from_kernel_info(kernel_info)

        self.assertEqual(mapping[3], "_Z29ncclDevFunc_AllGather_RING_LLv")
        self.assertEqual(mapping[4], "_Z32ncclDevFunc_AllGather_RING_LL128v")
        self.assertEqual(mapping[5], "_Z33ncclDevFunc_AllGather_RING_SIMPLEv")

    def test_nccl_devfunc_rewrites_fallback_to_multiway_dispatch(self):
        ptx = """
    cvt.u32.u16 %r289, %rs6;
    mul.wide.u32 %rd207, %r289, 8;
    mov.u64 %rd208, ncclDevFuncTable;
    add.s64 %rd209, %rd208, %rd207;
    ld.global.u64 %rd210, [%rd209];
    {
    .reg .b32 temp_param_reg;
    call.uni
    %rd210,
    (
    );
    } //
"""

        rewritten, rewrites = nccl_devfunc_prober.rewrite_table_fallback_to_local_dispatch(
            ptx,
            {
                3: "_Z29ncclDevFunc_AllGather_RING_LLv",
                4: "_Z32ncclDevFunc_AllGather_RING_LL128v",
            },
        )

        self.assertEqual(rewrites, 1)
        self.assertIn("setp.eq.s16 %ndft_p0_1, %rs6, 3;", rewritten)
        self.assertIn("setp.eq.s16 %ndft_p0_1, %rs6, 4;", rewritten)
        self.assertIn("_Z29ncclDevFunc_AllGather_RING_LLv", rewritten)
        self.assertIn("_Z32ncclDevFunc_AllGather_RING_LL128v", rewritten)
        self.assertIn("// begin rewritten ncclDevFuncTable fallback", rewritten)
        self.assertNotIn("FabricPerf static ncclDevFuncTable replacement", rewritten)
