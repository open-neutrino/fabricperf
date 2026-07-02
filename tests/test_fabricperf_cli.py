"""Tests for FabricPerf CLI mode selection and environment setup.

Functionality: cover public mode contracts, default probe selection, build
configuration, and launcher environment normalization. Example: throughput mode
sets `FABRICPERF_THROUGHPUT_PARTITIONS=4`.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from neutrino.plugins.fabricperf.tests._fabricperf_test_utils import (
    fabricperf_args,
    fabricperf_build_module,
)

from neutrino.cli import _is_toml_probe_path
from neutrino.plugins.fabricperf import analyze
from neutrino.plugins.fabricperf.analysis import common as analysis_common
from neutrino.plugins.fabricperf.analysis import latency as latency_analysis
from neutrino.plugins.fabricperf.analysis import memory as memory_analysis
from neutrino.plugins.fabricperf.analysis import throughput as throughput_analysis
from neutrino.plugins.fabricperf.cli import (
    configure_env,
    default_probe,
    deepep_enabled,
    fabricperf_mode,
    nccl_devfunc_enabled,
    post_run,
    prepare,
    throughput_partitions,
    validate_args,
)
from neutrino.plugins.fabricperf.nccl_devfunc import table as nccl_devfunc_table


class FabricPerfCliTests(unittest.TestCase):
    """Cover FabricPerf public latency/memory/throughput mode contract."""

    def test_mode_defaults_to_latency(self):
        self.assertEqual(fabricperf_mode({}), "latency")
        self.assertEqual(analysis_common.fabricperf_mode({}), "latency")

    def test_memory_mode_is_valid(self):
        env = {"FABRICPERF_MODE": "memory"}
        self.assertEqual(fabricperf_mode(env), "memory")
        self.assertEqual(analysis_common.fabricperf_mode(env), "memory")

    def test_throughput_modes_are_valid(self):
        env = {"FABRICPERF_MODE": "throughput"}
        self.assertEqual(fabricperf_mode(env), "throughput")
        self.assertEqual(analysis_common.fabricperf_mode(env), "throughput")
        self.assertEqual(analysis_common.throughput_variant(env), "gmem")

    def test_invalid_mode_fails(self):
        with self.assertRaises(ValueError):
            fabricperf_mode({"FABRICPERF_MODE": "range"})
        with self.assertRaises(ValueError):
            analysis_common.fabricperf_mode({"FABRICPERF_MODE": "app-replay"})
        with self.assertRaises(ValueError):
            fabricperf_mode({"FABRICPERF_MODE": "throughput_gmem"})
        with self.assertRaises(ValueError):
            fabricperf_mode({"FABRICPERF_MODE": "throughput_smem"})

    def test_analyze_reexports_split_analyzer_api(self):
        self.assertIs(analyze.fabricperf_mode, analysis_common.fabricperf_mode)
        self.assertIs(analyze.TraceStats, latency_analysis.TraceStats)
        self.assertIs(analyze.merge_one_pass_probe_output, memory_analysis.merge_one_pass_probe_output)
        self.assertIs(analyze.read_throughput_root, throughput_analysis.read_throughput_root)

    def test_default_probe_follows_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = {"dir": tmpdir}
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(default_probe(plugin, fabricperf_args()), str(Path(tmpdir) / "latency.probe"))
            with mock.patch.dict(os.environ, {"FABRICPERF_MODE": "memory"}, clear=True):
                self.assertEqual(default_probe(plugin, fabricperf_args()), str(Path(tmpdir) / "memory.probe"))
            with mock.patch.dict(os.environ, {"FABRICPERF_MODE": "throughput"}, clear=True):
                self.assertEqual(default_probe(plugin, fabricperf_args()), str(Path(tmpdir) / "throughput.probe"))

    def test_probe_suffix_loads_as_toml(self):
        self.assertTrue(_is_toml_probe_path("memory.probe"))
        self.assertTrue(_is_toml_probe_path("throughput.probe"))
        self.assertTrue(_is_toml_probe_path("throughput_smem_archive.probe"))
        self.assertTrue(_is_toml_probe_path("latency.toml"))
        self.assertFalse(_is_toml_probe_path("read.py"))

    def test_prepare_builds_latency_without_cupti(self):
        plugin = {"dir": "/tmp/fabricperf", "shared_object": "/tmp/fabricperf/build/fabricperf_plugin.so"}
        env = {"FABRICPERF_MODE": "latency"}
        build_module = fabricperf_build_module()
        with mock.patch("neutrino.plugins.fabricperf.cli._load_build_module", return_value=build_module):
            self.assertEqual(prepare(plugin, fabricperf_args(), env), 0)
        build_module.build_plugin.assert_called_once_with(
            "/tmp/fabricperf",
            shared_object="/tmp/fabricperf/build/fabricperf_plugin.so",
            mode="latency",
            cupti=False,
        )

    def test_prepare_builds_memory_with_cupti(self):
        plugin = {"dir": "/tmp/fabricperf", "shared_object": "/tmp/fabricperf/build/fabricperf_plugin.so"}
        env = {"FABRICPERF_MODE": "memory"}
        build_module = fabricperf_build_module()
        with mock.patch("neutrino.plugins.fabricperf.cli._load_build_module", return_value=build_module):
            self.assertEqual(prepare(plugin, fabricperf_args(), env), 0)
        build_module.build_plugin.assert_called_once_with(
            "/tmp/fabricperf",
            shared_object="/tmp/fabricperf/build/fabricperf_plugin.so",
            mode="memory",
            cupti=True,
        )

    def test_prepare_builds_throughput_without_cupti(self):
        plugin = {"dir": "/tmp/fabricperf", "shared_object": "/tmp/fabricperf/build/fabricperf_plugin.so"}
        env = {"FABRICPERF_MODE": "throughput"}
        build_module = fabricperf_build_module()
        with mock.patch("neutrino.plugins.fabricperf.cli._load_build_module", return_value=build_module):
            self.assertEqual(prepare(plugin, fabricperf_args(), env), 0)
        build_module.build_plugin.assert_called_once_with(
            "/tmp/fabricperf",
            shared_object="/tmp/fabricperf/build/fabricperf_plugin.so",
            mode="throughput",
            cupti=False,
        )

    def test_prepare_can_select_nccl_devfunc_prober(self):
        plugin = {"dir": "/tmp/fabricperf", "shared_object": "/tmp/fabricperf/build/fabricperf_plugin.so"}
        env = {"FABRICPERF_MODE": "throughput", "FABRICPERF_NCCL_DEVFUNC": "1"}
        with mock.patch("neutrino.plugins.fabricperf.cli._load_build_module", return_value=fabricperf_build_module()):
            self.assertEqual(prepare(plugin, fabricperf_args(), env), 0)
        self.assertEqual(env["NEUTRINO_PROBING_PY"], "/tmp/fabricperf/nccl/prober.py")

    def test_prepare_can_select_deepep_prober(self):
        plugin = {"dir": "/tmp/fabricperf", "shared_object": "/tmp/fabricperf/build/fabricperf_plugin.so"}
        env = {"FABRICPERF_MODE": "latency", "FABRICPERF_DEEPEP": "1"}
        with mock.patch("neutrino.plugins.fabricperf.cli._load_build_module", return_value=fabricperf_build_module()):
            self.assertEqual(prepare(plugin, fabricperf_args(), env), 0)
        self.assertEqual(env["NEUTRINO_PROBING_PY"], "/tmp/fabricperf/deepep/prober.py")

    def test_prepare_rejects_deepep_and_nccl_devfunc_together(self):
        plugin = {"dir": "/tmp/fabricperf", "shared_object": "/tmp/fabricperf/build/fabricperf_plugin.so"}
        env = {
            "FABRICPERF_MODE": "latency",
            "FABRICPERF_DEEPEP": "1",
            "FABRICPERF_NCCL_DEVFUNC": "1",
        }

        self.assertEqual(prepare(plugin, fabricperf_args(), env), 1)
        self.assertNotIn("NEUTRINO_PROBING_PY", env)

    def test_nccl_devfunc_env_is_opt_in(self):
        self.assertFalse(nccl_devfunc_enabled({}))
        self.assertFalse(nccl_devfunc_table.enabled({}))
        self.assertTrue(nccl_devfunc_enabled({"FABRICPERF_NCCL_DEVFUNC": "true"}))
        self.assertTrue(nccl_devfunc_table.enabled({"FABRICPERF_NCCL_DEVFUNC": "1"}))
        self.assertFalse(deepep_enabled({}))
        self.assertTrue(deepep_enabled({"FABRICPERF_DEEPEP": "yes"}))

    def test_validate_rejects_old_fabricperf_workflows(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                validate_args(fabricperf_args(no_probe=True))
            with self.assertRaises(ValueError):
                validate_args(fabricperf_args(app_replay=True))

    def test_configure_env_normalizes_mode(self):
        env = {"FABRICPERF_MODE": "MEMORY"}
        configure_env(fabricperf_args(), env)
        self.assertEqual(env["FABRICPERF_MODE"], "memory")
        self.assertNotIn("FABRICPERF_THROUGHPUT_PARTITIONS", env)

    def test_configure_env_sets_throughput_partitions(self):
        env = {"FABRICPERF_MODE": "THROUGHPUT"}
        configure_env(fabricperf_args(), env)
        self.assertEqual(env["FABRICPERF_MODE"], "throughput")
        self.assertEqual(env["FABRICPERF_THROUGHPUT_PARTITIONS"], "4")

    def test_throughput_partitions_are_validated(self):
        self.assertEqual(throughput_partitions({}), "4")
        self.assertEqual(throughput_partitions({"FABRICPERF_THROUGHPUT_PARTITIONS": "4"}), "4")
        self.assertEqual(analysis_common.throughput_partitions({"FABRICPERF_THROUGHPUT_PARTITIONS": "4"}), 4)
        with self.assertRaises(ValueError):
            throughput_partitions({"FABRICPERF_THROUGHPUT_PARTITIONS": "2"})
        with self.assertRaises(ValueError):
            analysis_common.throughput_partitions({"FABRICPERF_THROUGHPUT_PARTITIONS": "1"})
