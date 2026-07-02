import unittest

from neutrino.probe.ptx_match import parse_ptx_lines, select_match


class FabricPerfNcclPtxMatchTests(unittest.TestCase):
    """Cover NCCL-specific PTX semantic matches owned by FabricPerf.

    Motivation: FabricPerf's NCCL devFunc wrapper depends on funcId dispatch
    capture. Example: SendRecv compares shared funcId register `%rs14` with
    immediate `669` before branching to the local fast path.
    """

    def test_nccl_dev_func_dispatch_auto_captures_func_id(self):
        lines = [
            "    setp.eq.s16 %p1, %rs2, 0;",
            "    setp.eq.s16 %p39, %rs14, 669;",
            "    @%p39 bra $L__BB0_42;",
        ]
        instructions = parse_ptx_lines(lines)

        match = select_match(
            instructions,
            {"kind": "nccl_dev_func_dispatch", "unique": True},
            {},
            {},
            "original_sendrecv_dispatch",
        )

        self.assertEqual(match.idx, 1)
        self.assertEqual(match.captures["func_pred"], "%p39")
        self.assertEqual(match.captures["func_id_reg"], "%rs14")
        self.assertEqual(match.captures["func_id_imm"], "669")

    def test_deepep_tma_wait_selects_last_wait_group(self):
        """Cover DeepEP's TMA wait anchor used by FabricPerf probes.

        Motivation: DeepEP V2 kernels expose TMA communication through
        `cp.async.bulk.wait_group 0`, not NCCL polling loops. Example: the
        throughput and latency probes select the last wait in a dispatch kernel.
        """
        lines = [
            "    cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%r8], [%rd180], %r591, [%r587], %rd181;",
            "    cp.async.bulk.wait_group 0;",
            "    cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%rd230], [%r8], %r591, %rd183;",
            "    cp.async.bulk.commit_group;",
            "    cp.async.bulk.wait_group 0;",
        ]
        instructions = parse_ptx_lines(lines)

        match = select_match(
            instructions,
            {"kind": "deepep_tma_wait", "last": True},
            {},
            {},
            "deepep_tma_done",
        )

        self.assertEqual(match.idx, 4)
        self.assertEqual(match.raw.strip(), "cp.async.bulk.wait_group 0;")


if __name__ == "__main__":
    unittest.main()
