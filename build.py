#!/usr/bin/env python3
"""Build the FabricPerf runtime shared object from the plugin directory.

Example:
    python3 neutrino/plugins/fabricperf/build.py --mode throughput
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


FABRICPERF_MODE_ENV = "FABRICPERF_MODE"
LATENCY_MODE = "latency"
MEMORY_MODE = "memory"
THROUGHPUT_MODE = "throughput"
VALID_MODES = (LATENCY_MODE, MEMORY_MODE, THROUGHPUT_MODE)


def fail(message: str) -> int:
    """Print one FabricPerf build error and return a shell failure.

    Example: fail("make not found") emits a concise diagnostic for `neutrino`
    and for direct `build.py` use.
    """
    print(f"[error] {message}", file=sys.stderr)
    return 1


def normalized_mode(raw_mode: str) -> str:
    """Return a validated FabricPerf build mode.

    Example: normalized_mode("MEMORY") returns "memory", matching the mode
    accepted by `Makefile`.
    """
    mode = raw_mode.strip().lower()
    if mode not in VALID_MODES:
        joined = ", ".join(VALID_MODES)
        raise ValueError(f"{FABRICPERF_MODE_ENV} must be one of: {joined}")
    return mode


def build_plugin(
    plugin_dir: str | os.PathLike[str],
    *,
    shared_object: str | os.PathLike[str] | None = None,
    mode: str = LATENCY_MODE,
    cupti: bool | None = None,
) -> int:
    """Build FabricPerf with the plugin-local Makefile.

    Functionality: this is the only FabricPerf runtime build path used by the
    Neutrino hook. Example: `mode="memory"` enables `FABRICPERF_CUPTI=1`.
    """
    plugin_path = Path(plugin_dir)
    build_mode = normalized_mode(mode)
    needs_cupti = build_mode == MEMORY_MODE if cupti is None else cupti
    output = Path(shared_object) if shared_object is not None else plugin_path / "build" / "fabricperf_plugin.so"

    make_env = os.environ.copy()
    make_env[FABRICPERF_MODE_ENV] = build_mode
    if needs_cupti:
        make_env["FABRICPERF_CUPTI"] = "1"
    else:
        make_env.pop("FABRICPERF_CUPTI", None)

    try:
        result = subprocess.run(["make", "-C", str(plugin_path)], env=make_env)
    except FileNotFoundError:
        return fail("make not found while building FabricPerf")
    except OSError as exc:
        return fail(f"failed to build FabricPerf: {exc}")

    if result.returncode != 0:
        return fail(f"FabricPerf build failed with status {result.returncode}")
    if not output.exists():
        return fail(f"FabricPerf build did not produce {output}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse direct FabricPerf build options.

    Example: `--mode memory --cupti` builds the CUPTI-enabled memory runtime.
    """
    parser = argparse.ArgumentParser(description="Build the FabricPerf plugin runtime")
    parser.add_argument("--plugin-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--shared-object", default=None)
    parser.add_argument("--mode", default=os.environ.get(FABRICPERF_MODE_ENV, LATENCY_MODE))
    parser.add_argument("--cupti", action="store_true", help="force FABRICPERF_CUPTI=1")
    parser.add_argument("--no-cupti", action="store_true", help="force FABRICPERF_CUPTI unset")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the direct FabricPerf build command.

    Example: `python3 build.py --mode throughput` compiles `throughput.c`.
    """
    args = parse_args(argv)
    cupti = None
    if args.cupti and args.no_cupti:
        return fail("--cupti and --no-cupti cannot both be set")
    if args.cupti:
        cupti = True
    elif args.no_cupti:
        cupti = False
    try:
        return build_plugin(
            args.plugin_dir,
            shared_object=args.shared_object,
            mode=args.mode,
            cupti=cupti,
        )
    except ValueError as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
