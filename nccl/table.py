"""FabricPerf-local NCCL device-function target selection.

This package deliberately does not modify Neutrino core. Example: the CUDA
prober wrapper uses this table to decide that `ncclDevKernel_SendRecv` should
prefer the local `_Z20ncclDevFunc_SendRecvv` body when it exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re


PTX_SYMBOL_NAME = r"[A-Za-z_$][A-Za-z0-9_$]*"
LOCAL_DEVFUNC_RE = re.compile(
    rf"(?m)^\s*\.visible\s+\.func\s+(?P<name>{PTX_SYMBOL_NAME}ncclDevFunc_{PTX_SYMBOL_NAME})\s*\("
)
SET_EQ_S16_IMM_RE = re.compile(
    r"(?m)^\s*setp\.eq\.s16\s+%p\d+,\s+%[A-Za-z0-9_]+,\s+(?P<index>\d+);"
)
KERNEL_DESCRIPTOR_RE = re.compile(
    r"ncclDevKernel_(?P<descriptor>[A-Za-z0-9_]+?)(?:\d+ncclDevKernelArgsStorage|$)"
)
DEVFUNC_DESCRIPTOR_RE = re.compile(
    r"ncclDevFunc_(?P<descriptor>[A-Za-z0-9_]+)v$"
)

ENV_ENABLE = "FABRICPERF_NCCL_DEVFUNC"


@dataclass(frozen=True)
class NcclDevFuncRule:
    """One kernel-name rule for selecting a local NCCL device function.

    Motivation: NCCL mangled names are long, but stable substrings are enough
    for FabricPerf's initial table. Example: `ncclDevKernel_SendRecv` maps to
    a function containing `ncclDevFunc_SendRecv`.
    """

    kernel_substring: str
    function_substring: str
    description: str


@dataclass(frozen=True)
class NcclDevFuncTarget:
    """Resolved NCCL target for one generated PTX module."""

    kernel_name: str
    function_name: str
    function_id: int | None
    source: str
    candidates: tuple[str, ...]


DEFAULT_RULES: tuple[NcclDevFuncRule, ...] = (
    NcclDevFuncRule(
        kernel_substring="ncclDevKernel_SendRecv",
        function_substring="ncclDevFunc_SendRecv",
        description="NCCL SendRecv wrapper dispatches to the SendRecv devFunc.",
    ),
)


def enabled(environ: dict[str, str] | None = None) -> bool:
    """Return whether FabricPerf should use the NCCL devFunc prober wrapper."""
    source = os.environ if environ is None else environ
    value = source.get(ENV_ENABLE, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def local_devfuncs(ptx: str) -> list[str]:
    """List local visible NCCL devFunc bodies in PTX order."""
    functions: list[str] = []
    seen: set[str] = set()
    for match in LOCAL_DEVFUNC_RE.finditer(ptx):
        name = match.group("name")
        if name in seen:
            continue
        functions.append(name)
        seen.add(name)
    return functions


def infer_single_func_id(entry_ptx: str) -> int | None:
    """Infer a single NCCL funcId from a wrapper dispatch body.

    Motivation: simple wrappers compare the runtime funcId against exactly one
    nonzero immediate. Example: SendRecv exposes `setp.eq.s16 ..., 669`.
    """
    candidates = {
        int(match.group("index"))
        for match in SET_EQ_S16_IMM_RE.finditer(entry_ptx)
        if int(match.group("index")) != 0
    }
    if len(candidates) != 1:
        return None
    return next(iter(candidates))


def kernel_descriptor(kernel_name: str) -> str | None:
    """Return the NCCL descriptor embedded in a mangled kernel name.

    Motivation: collective kernels and device functions share a stable textual
    descriptor even when the C++ mangling length changes. Example:
    `_Z31ncclDevKernel_Broadcast_RING_LL24...` maps to `Broadcast_RING_LL`.
    """
    match = KERNEL_DESCRIPTOR_RE.search(kernel_name)
    if match is None:
        return None
    return match.group("descriptor")


def devfunc_descriptor(function_name: str) -> str | None:
    """Return the NCCL descriptor embedded in a mangled devFunc name.

    Motivation: exact descriptor matching avoids treating `RING_LL128` as a
    match for `RING_LL`. Example: `_Z29ncclDevFunc_Broadcast_RING_LLv` maps to
    `Broadcast_RING_LL`.
    """
    match = DEVFUNC_DESCRIPTOR_RE.search(function_name)
    if match is None:
        return None
    return match.group("descriptor")


def descriptor_matches(kernel_name: str, functions: list[str]) -> list[str]:
    """Find local devFuncs whose descriptor exactly matches the kernel."""
    descriptor = kernel_descriptor(kernel_name)
    if descriptor is None:
        return []
    return [name for name in functions if devfunc_descriptor(name) == descriptor]


def select_target(kernel_name: str,
                  ptx: str,
                  entry_ptx: str,
                  environ: dict[str, str] | None = None) -> NcclDevFuncTarget | None:
    """Select the FabricPerf NCCL devFunc target for one kernel.

    Selection order:
    1. exact `ncclDevKernel_<descriptor>` to `ncclDevFunc_<descriptor>` match
    2. FabricPerf's built-in matching table
    3. the only local `ncclDevFunc_*` body, if exactly one exists
    """
    _ = environ
    functions = local_devfuncs(ptx)
    if len(functions) == 0:
        return None

    function_id = infer_single_func_id(entry_ptx)

    matches = descriptor_matches(kernel_name, functions)
    if len(matches) == 1:
        return NcclDevFuncTarget(kernel_name, matches[0], function_id, "kernel-descriptor-match", tuple(functions))

    for rule in DEFAULT_RULES:
        if rule.kernel_substring not in kernel_name:
            continue
        matches = [name for name in functions if rule.function_substring in name]
        if len(matches) == 1:
            return NcclDevFuncTarget(kernel_name, matches[0], function_id, rule.description, tuple(functions))

    if len(functions) == 1:
        return NcclDevFuncTarget(kernel_name, functions[0], function_id, "single-local-devfunc", tuple(functions))
    return None
