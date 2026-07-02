"""FabricPerf NCCL rewrite tests for Neutrino CUDA probe helpers.

Example:
    python3 -m unittest neutrino.plugins.fabricperf.tests.test_cuda_nccl_rewrite
"""

import unittest
import tempfile

from neutrino.probe import KernelParam
from neutrino.probe.cuda import (
    append_nccl_runtime_globals,
    discover_ptx_entries,
    inject_nccl_runtime_discovery,
    kernel_filter_reason,
    nccl_dev_func_table_local_map,
    nccl_runtime_patch_info,
    nccl_runtime_patch_kernel,
    parse_global_array_definitions,
    rewrite_nccl_dev_func_table_fallback,
    select_ptx_module,
    write_kernel_info,
)


class CudaNcclRewriteTests(unittest.TestCase):
    def test_discovers_ptx_entries_from_concatenated_dump(self):
        # Example: `cuda.py --all` sees cuobjdump output with multiple PTX modules.
        dump = """
Fatbin ptx code:
.version 8.7
.target sm_90
.visible .entry first_kernel()
{
    ret;
}

Fatbin ptx code:
.version 8.7
.target sm_90
.entry second_kernel()
{
    ret;
}
"""

        self.assertEqual(discover_ptx_entries(dump), ["first_kernel", "second_kernel"])

    def test_kernel_filter_reason_reports_skip_reason(self):
        # Example: single mode exits on this reason, while `--all` writes a skipped row.
        self.assertIsNone(kernel_filter_reason("ncclDevKernel_SendRecv", None, ["SendRecv"], None))
        self.assertIn(
            "not in",
            kernel_filter_reason("ncclDevKernel_SendRecv", None, ["Broadcast"], None),
        )
        self.assertIn(
            "filtered out",
            kernel_filter_reason("ncclDevKernel_SendRecv", ["SendRecv"], None, None),
        )
        self.assertIn(
            "prefix-filtered",
            kernel_filter_reason("_Z13ncclDevKernelv", None, None, ["ncclDevKernel"]),
        )

    def test_select_ptx_module_strips_single_cuobjdump_banner(self):
        # Example: NCCL D8 sendrecv can dump one PTX module preceded by a Fatbin banner.
        dump = """
Fatbin ptx code:
================
arch = sm_90

.version 8.7
.target sm_90
.address_size 64

.visible .entry kernel()
{
    ret;
}
"""

        selected = select_ptx_module(dump, "kernel")

        self.assertTrue(selected.startswith(".version 8.7"))
        self.assertNotIn("Fatbin ptx code", selected)

    def test_maps_local_nccl_dev_funcs_through_linked_initializer(self):
        full_ptx = """
.version 8.7
.visible .global .align 8 .u64 ncclDevFuncTable[6] = {
    _Z20ncclDevFunc_SendRecvv,
    _Z29ncclDevFunc_Broadcast_RING_LLv,
    _Z32ncclDevFunc_Broadcast_RING_LL128v,
    _Z33ncclDevFunc_Broadcast_RING_SIMPLEv,
    0,
    _Z34ncclDevFunc_Reduce_Sum_f32_RING_LLv
};
"""
        selected_module = """
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
        definition = parse_global_array_definitions(full_ptx)["ncclDevFuncTable"]

        mapping = nccl_dev_func_table_local_map(definition, selected_module)

        self.assertEqual(
            mapping,
            {
                1: "_Z29ncclDevFunc_Broadcast_RING_LLv",
                2: "_Z32ncclDevFunc_Broadcast_RING_LL128v",
                3: "_Z33ncclDevFunc_Broadcast_RING_SIMPLEv",
            },
        )

    def test_rewrites_indirect_nccl_dev_func_table_call_to_direct_branches(self):
        entry_ptx = """
.visible .entry kernel()
{
    ld.shared.u16 %rs6, [ncclShmem+522];
    cvt.u32.u16 %r289, %rs6;
    mul.wide.u32 %rd207, %r289, 8;
    mov.u64 %rd208, ncclDevFuncTable;
    add.s64 %rd209, %rd208, %rd207;
    ld.global.u64 %rd210, [%rd209];
{ //
.reg .b32 temp_param_reg;
prototype_1 : .callprototype ()_ ();
call
%rd210,
(
)
, prototype_1;
} //
    ret;
}
"""
        mapping = {
            342: "_Z29ncclDevFunc_Broadcast_RING_LLv",
            343: "_Z32ncclDevFunc_Broadcast_RING_LL128v",
        }

        rewritten, replacements = rewrite_nccl_dev_func_table_fallback(entry_ptx, mapping)

        self.assertEqual(replacements, 1)
        self.assertIn("setp.eq.s16 %ndft_p0_1, %rs6, 342;", rewritten)
        self.assertIn("_Z29ncclDevFunc_Broadcast_RING_LLv,", rewritten)
        self.assertIn("_Z32ncclDevFunc_Broadcast_RING_LL128v,", rewritten)
        self.assertIn("trap;", rewritten)
        self.assertNotIn("mov.u64 %rd208, ncclDevFuncTable;", rewritten)
        self.assertNotIn("%rd210,", rewritten)

    def test_generates_runtime_patch_kernel_for_local_dev_funcs(self):
        definition = {
            "storage": "global",
            "dtype": "u64",
            "count": "512",
        }
        mapping = {
            342: "_Z29ncclDevFunc_Broadcast_RING_LLv",
            343: "_Z32ncclDevFunc_Broadcast_RING_LL128v",
        }

        info = nccl_runtime_patch_info(definition, mapping)
        generated = nccl_runtime_patch_kernel(info)

        self.assertIn(".visible .entry __neutrino_patch_ncclDevFuncTable", generated)
        self.assertIn("setp.eq.u32 %nrt_patch_p1, %nrt_patch_r1, 342;", generated)
        self.assertIn("mov.u64 %nrt_patch_rd2, _Z29ncclDevFunc_Broadcast_RING_LLv;", generated)
        self.assertIn("st.global.u64 [%nrt_patch_rd4], %nrt_patch_rd2;", generated)

    def test_runtime_discovery_keeps_original_table_fallback(self):
        entry_ptx = """
.visible .entry kernel()
{
    ld.shared.u16 %rs6, [ncclShmem+522];
    setp.eq.s16 %p39, %rs6, 342;
    @%p39 bra $L__direct;
    cvt.u32.u16 %r289, %rs6;
    mul.wide.u32 %rd207, %r289, 8;
    mov.u64 %rd208, ncclDevFuncTable;
    add.s64 %rd209, %rd208, %rd207;
    ld.global.u64 %rd210, [%rd209];
    ret;
}
"""
        info = nccl_runtime_patch_info(
            {"storage": "global", "dtype": "u64", "count": "512"},
            {342: "_Z29ncclDevFunc_Broadcast_RING_LLv"},
        )

        rewritten, guards = inject_nccl_runtime_discovery(entry_ptx, info)

        self.assertEqual(guards, 1)
        self.assertIn("__neutrino_nccl_discovery_mode", rewritten)
        self.assertIn("__neutrino_nccl_runtime_func_id", rewritten)
        self.assertIn("cvt.u64.u16 %nrt_discover_rd3, %rs6;", rewritten)
        self.assertIn("mov.u64 %rd208, ncclDevFuncTable;", rewritten)

    def test_kernel_info_appends_optional_runtime_table_section(self):
        info = nccl_runtime_patch_info(
            {"storage": "global", "dtype": "u64", "count": "512"},
            {344: "_Z33ncclDevFunc_Broadcast_RING_SIMPLEv"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            write_kernel_info(
                "kernel",
                [KernelParam("u64", "kernel_param_0")],
                [],
                [],
                tmpdir,
                nccl_runtime=info,
            )
            with open(f"{tmpdir}/kernel.info", "r") as handle:
                content = handle.read()

        self.assertIn("NCCL_RUNTIME_TABLE", content)
        self.assertIn("__neutrino_patch_ncclDevFuncTable", content)
        self.assertIn("344,_Z33ncclDevFunc_Broadcast_RING_SIMPLEv", content)

    def test_runtime_globals_are_appended_once(self):
        info = nccl_runtime_patch_info(
            {"storage": "global", "dtype": "u64", "count": "512"},
            {342: "_Z29ncclDevFunc_Broadcast_RING_LLv"},
        )

        first = append_nccl_runtime_globals(".version 8.7\n", info)
        second = append_nccl_runtime_globals(first, info)

        self.assertEqual(second.count("__neutrino_nccl_discovery_mode"), 1)
        self.assertEqual(second.count("__neutrino_nccl_runtime_func_id"), 1)


if __name__ == "__main__":
    unittest.main()
