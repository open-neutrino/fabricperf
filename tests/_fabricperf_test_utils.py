"""Shared helpers for FabricPerf plugin tests.

Functionality: keep lightweight parser stubs and argparse-like fixtures in one
place so functionality-specific test modules can stay small. Example:
`fabricperf_args(no_probe=True)` mimics the CLI namespace used by validate_args.
"""

import sys
import types
from unittest import mock

# Step: tests do not need a real TOML parser when only path selection is used.
sys.modules.setdefault("toml", types.SimpleNamespace())


def fabricperf_args(**overrides):
    """Build the minimal argparse-like object used by FabricPerf CLI hooks.

    Motivation: unit tests should exercise plugin behavior without invoking the
    full Neutrino parser. Example: fabricperf_args(no_probe=True) should be
    rejected by validate_args.
    """
    values = {"no_probe": False, "app_replay": False}
    values.update(overrides)
    return types.SimpleNamespace(**values)


def fabricperf_build_module(return_value=0):
    """Build a mock FabricPerf build module for CLI hook tests.

    Example: prepare() should call build_plugin("/tmp/fabricperf", mode="memory").
    """
    return types.SimpleNamespace(build_plugin=mock.Mock(return_value=return_value))
