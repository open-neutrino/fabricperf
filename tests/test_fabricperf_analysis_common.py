"""FabricPerf analysis helper tests."""

import tempfile
import unittest
from pathlib import Path

from neutrino.plugins.fabricperf.analysis.common import parse_rank


class FabricPerfAnalysisCommonTests(unittest.TestCase):
    """Cover trace-log parsing shared by FabricPerf analyzers."""

    def test_parse_rank_accepts_non_mpi_exchange_log(self):
        """Non-MPI DeepEP torch workers should still expose rank metadata."""
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp)
            (trace_dir / "event.log").write_text(
                "[plugin][fabricperf] prepared non-MPI rank=6 world=8 exchange=/tmp/fp\n",
                encoding="utf-8",
            )

            self.assertEqual(parse_rank(trace_dir), 6)


if __name__ == "__main__":
    unittest.main()
