"""Standalone FabricPerf bridge to the local Neutrino TOML shim.

Functionality: let this workspace run without the external `toml` package when
the paired Neutrino checkout is present. Example: `import toml; toml.load(...)`
uses `../neutrino/toml.py`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, TextIO


_SHIM_PATH = Path(__file__).resolve().parents[1] / "neutrino" / "toml.py"
if not _SHIM_PATH.exists():
    raise ModuleNotFoundError("install the Python toml package or keep the paired Neutrino checkout")

_SPEC = importlib.util.spec_from_file_location("_fabricperf_neutrino_toml_shim", _SHIM_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ModuleNotFoundError(f"could not load TOML shim: {_SHIM_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def load(source: str | Path | TextIO) -> dict[str, Any]:
    """Load TOML through the paired Neutrino shim."""
    return _MODULE.load(source)


def loads(text: str) -> dict[str, Any]:
    """Parse TOML through the paired Neutrino shim."""
    return _MODULE.loads(text)


def dump(data: dict[str, Any], destination: TextIO) -> None:
    """Write TOML through the paired Neutrino shim."""
    _MODULE.dump(data, destination)


def dumps(data: dict[str, Any]) -> str:
    """Serialize TOML through the paired Neutrino shim."""
    return _MODULE.dumps(data)
