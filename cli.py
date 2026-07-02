"""CLI integration hooks for the FabricPerf built-in plugin."""

from __future__ import annotations

import os
import importlib.util
import subprocess
import sys


FABRICPERF_MODE_ENV = "FABRICPERF_MODE"
FABRICPERF_THROUGHPUT_PARTITIONS_ENV = "FABRICPERF_THROUGHPUT_PARTITIONS"
FABRICPERF_NCCL_DEVFUNC_ENV = "FABRICPERF_NCCL_DEVFUNC"
FABRICPERF_DEEPEP_ENV = "FABRICPERF_DEEPEP"
LATENCY_MODE = "latency"
MEMORY_MODE = "memory"
THROUGHPUT_MODE = "throughput"
VALID_MODES = (LATENCY_MODE, MEMORY_MODE, THROUGHPUT_MODE)
VALID_THROUGHPUT_PARTITIONS = ("4",)
DEFAULT_THROUGHPUT_PARTITIONS = "4"


def _fail(message: str) -> int:
    """Return a shell failure after printing one FabricPerf-owned error.

    Motivation: plugin orchestration runs outside argparse, so failures need a
    simple status code. Example: _fail("make not found") prints to stderr and
    returns 1.
    """
    print(f"[error] {message}", file=sys.stderr)
    return 1


def fabricperf_mode(environ: dict[str, str] | None = None) -> str:
    """Return the selected public FabricPerf mode.

    Motivation: FabricPerf now exposes one high-level selector instead of CUPTI
    backend knobs. Example: FABRICPERF_MODE=memory selects memory.probe and PM
    Sampling, while an unset value selects latency.probe.
    """
    source = os.environ if environ is None else environ
    mode = source.get(FABRICPERF_MODE_ENV, LATENCY_MODE).strip().lower()
    if mode not in VALID_MODES:
        joined = ", ".join(VALID_MODES)
        raise ValueError(f"{FABRICPERF_MODE_ENV} must be one of: {joined}")
    return mode


def throughput_partitions(environ: dict[str, str] | None = None) -> str:
    """Return the requested throughput workgroup partition count.

    Motivation: partition count is part of the throughput trace contract even
    though the current probe layout only supports four slices. Example:
    `FABRICPERF_THROUGHPUT_PARTITIONS=4` is accepted and propagated to the
    runtime/analyzer.
    """
    source = os.environ if environ is None else environ
    raw = source.get(FABRICPERF_THROUGHPUT_PARTITIONS_ENV, DEFAULT_THROUGHPUT_PARTITIONS).strip()
    if raw not in VALID_THROUGHPUT_PARTITIONS:
        joined = ", ".join(VALID_THROUGHPUT_PARTITIONS)
        raise ValueError(f"{FABRICPERF_THROUGHPUT_PARTITIONS_ENV} must be one of: {joined}")
    return raw


def nccl_devfunc_enabled(environ: dict[str, str] | None = None) -> bool:
    """Return whether FabricPerf should replace Neutrino's CUDA prober.

    Motivation: selected NCCL devFunc probing is experimental and plugin-owned.
    Example: `FABRICPERF_NCCL_DEVFUNC=1` makes prepare() point
    `NEUTRINO_PROBING_PY` at FabricPerf's wrapper.
    """
    source = os.environ if environ is None else environ
    value = source.get(FABRICPERF_NCCL_DEVFUNC_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def deepep_enabled(environ: dict[str, str] | None = None) -> bool:
    """Return whether FabricPerf should specialize probes for DeepEP kernels.

    Motivation: DeepEP V2 uses JIT-generated TMA kernels rather than NCCL
    `ncclDevFunc_*` bodies. Example: `FABRICPERF_DEEPEP=1` points
    `NEUTRINO_PROBING_PY` at FabricPerf's DeepEP prober wrapper.
    """
    source = os.environ if environ is None else environ
    value = source.get(FABRICPERF_DEEPEP_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def default_probe(plugin: dict[str, str], args) -> str:
    """Return FabricPerf's mode-specific default probe path.

    Motivation: core Neutrino only asks for a default probe; FabricPerf owns
    whether the run is latency, memory, or throughput. Example: direct-GMEM
    throughput mode returns plugins/fabricperf/throughput.probe.
    """
    _ = args
    mode = fabricperf_mode()
    if mode == MEMORY_MODE:
        filename = "memory.probe"
    elif mode == THROUGHPUT_MODE:
        filename = "throughput.probe"
    else:
        filename = "latency.probe"
    return os.path.join(plugin["dir"], filename)


def validate_args(args) -> None:
    """Reject core CLI combinations FabricPerf no longer supports.

    Motivation: Range Profiling and CUPTI-only app replay were removed from
    FabricPerf. Example: `--plugin fabricperf --no-probe` now fails before the
    workload launches.
    """
    mode = fabricperf_mode()
    if mode == THROUGHPUT_MODE:
        throughput_partitions()
    if nccl_devfunc_enabled() and deepep_enabled():
        raise ValueError(f"{FABRICPERF_NCCL_DEVFUNC_ENV} and {FABRICPERF_DEEPEP_ENV} cannot both be enabled")
    if args.no_probe:
        raise ValueError("FabricPerf requires a Neutrino probe; use FABRICPERF_MODE=memory for memory collection")
    if args.app_replay:
        raise ValueError("FabricPerf no longer supports --app-replay")


def _load_build_module(plugin: dict[str, str]):
    """Load FabricPerf's plugin-local build module.

    Motivation: runtime compilation is FabricPerf-owned, not setup.py-owned.
    Example: prepare() imports `build.py` and calls build_plugin().
    """
    build_path = os.path.join(plugin["dir"], "build.py")
    spec = importlib.util.spec_from_file_location("_neutrino_builtin_plugin_fabricperf_build", build_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load FabricPerf build module: {build_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_env(args, env: dict[str, str]) -> None:
    """Apply the selected FabricPerf mode to a launch environment.

    Motivation: target and analyzer subprocesses should see one normalized mode.
    Example: an unset environment becomes FABRICPERF_MODE=latency.
    """
    _ = args
    mode = fabricperf_mode(env)
    env[FABRICPERF_MODE_ENV] = mode
    if mode == THROUGHPUT_MODE:
        env[FABRICPERF_THROUGHPUT_PARTITIONS_ENV] = throughput_partitions(env)
    else:
        env.pop(FABRICPERF_THROUGHPUT_PARTITIONS_ENV, None)


def prepare(plugin: dict[str, str], args, env: dict[str, str]) -> int:
    """Prepare FabricPerf before the workload launches.

    Motivation: the core CLI only knows that a plugin has a prepare hook.
    Example: memory mode compiles the CUPTI PM Sampling runtime.
    """
    _ = args
    mode = fabricperf_mode(env)
    devfunc = nccl_devfunc_enabled(env)
    deepep = deepep_enabled(env)
    if devfunc and deepep:
        return _fail(f"{FABRICPERF_NCCL_DEVFUNC_ENV} and {FABRICPERF_DEEPEP_ENV} cannot both be enabled")
    if devfunc:
        # Override after core CLI selects CUDA's default prober. Example:
        # Neutrino still launches its hook, but FabricPerf owns devFunc PTX
        # post-processing inside the generated kernel workdir.
        env["NEUTRINO_PROBING_PY"] = os.path.join(plugin["dir"], "nccl", "prober.py")
    elif deepep:
        # Override after core CLI selects CUDA's default prober. Example:
        # DeepEP kernels keep the core PTX rewrite path but use TMA-specific
        # FabricPerf probe anchors in the plugin-owned wrapper.
        env["NEUTRINO_PROBING_PY"] = os.path.join(plugin["dir"], "deepep", "prober.py")
    try:
        build_module = _load_build_module(plugin)
    except (ImportError, OSError) as exc:
        return _fail(str(exc))
    return build_module.build_plugin(
        plugin["dir"],
        shared_object=plugin["shared_object"],
        mode=mode,
        cupti=mode == MEMORY_MODE,
    )


def post_run(plugin: dict[str, str], args, tracedir: str, python: str) -> None:
    """Run FabricPerf analysis after probe-backed workloads complete.

    Motivation: post-processing should be plugin-owned and should not replace the
    workload exit code. Example: latency mode prints tables, while memory mode
    merges probe bytes into PM Sampling CSV rows.
    """
    _ = args
    analyzer_env = os.environ.copy()
    mode = fabricperf_mode()
    analyzer_env[FABRICPERF_MODE_ENV] = mode
    if mode == THROUGHPUT_MODE:
        analyzer_env[FABRICPERF_THROUGHPUT_PARTITIONS_ENV] = throughput_partitions(analyzer_env)
    analyzer = plugin["analyzer"]
    try:
        result = subprocess.run([python, analyzer, tracedir], env=analyzer_env)
    except OSError as exc:
        print(f"[warn] FabricPerf analysis failed to start: {exc}", file=sys.stderr)
        return
    if result.returncode != 0:
        print(f"[warn] FabricPerf analysis exited with status {result.returncode}", file=sys.stderr)
