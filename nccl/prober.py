"""FabricPerf CUDA prober wrapper for selected NCCL devFunc instrumentation.

Usage is intentionally compatible with Neutrino's CUDA prober:

    nccl/prober.py <workdir> <kernel_name> [probe.toml]

The wrapper first calls `neutrino/probe/cuda.py` with a shell probe so Neutrino
still owns map metadata, symbol metadata, and module loading contracts. It then
rewrites `probed.ptx` inside the generated workdir to instrument the selected
local `ncclDevFunc_*` body and reassembles `probed.bin`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import copy
import re
import subprocess
import sys
import toml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from neutrino.common import load
from neutrino.probe import cuda as cuda_probe
from neutrino.plugins.fabricperf.nccl import table as nccl_devfunc_table


PTX_SYMBOL_NAME = r"[A-Za-z_$][A-Za-z0-9_$]*"
PTXAS_OPTION_RE = re.compile(r"(?P<option>-maxrregcount(?:=|\s+)\d+)")
PTX_GENERATED_SYMBOL_RE = re.compile(r"\b(__neutrino_map_(?:ptr|count)_\d+)\b")
NCCL_TABLE_CVT_RE = re.compile(r"^\s*cvt\.u32\.u16\s+(?P<dst>%r\d+),\s*(?P<src>%rs\d+);")
NCCL_TABLE_MUL_RE = re.compile(r"^\s*mul\.wide\.u32\s+(?P<dst>%rd\d+),\s*(?P<src>%r\d+),\s*8;")
NCCL_TABLE_MOV_RE = re.compile(r"^\s*mov\.u64\s+(?P<dst>%rd\d+),\s*ncclDevFuncTable;")
NCCL_TABLE_ADD_RE = re.compile(
    r"^\s*add\.s64\s+(?P<dst>%rd\d+),\s*(?P<table>%rd\d+),\s*(?P<offset>%rd\d+);"
)
NCCL_TABLE_LD_RE = re.compile(r"^\s*ld\.global\.u64\s+(?P<dst>%rd\d+),\s*\[(?P<addr>%rd\d+)\];")
PTX_FUNCTION_OR_ENTRY_RE = re.compile(r"^\s*(?:\.visible\s+)?\.(?:func|entry)\b")
NCCL_DIRECT_DISPATCH_SET_RE = re.compile(
    r"^\s*setp\.(?P<op>eq|ne)\.s16\s+(?P<pred>%p\d+),\s+(?P<src>%rs\d+),\s*(?P<func_id>\d+);"
)
NCCL_PREDICATED_BRA_RE = re.compile(r"^(?P<indent>\s*)@(?P<pred>%p\d+)\s+bra\s+(?P<target>[^;]+);")
NCCL_UNCONDITIONAL_BRA_RE = re.compile(r"^\s*bra\.uni\s+(?P<target>[^;]+);")
THROUGHPUT_RECV_PROBE = "fabricperf_throughput_recv"
THROUGHPUT_FLUSH_PROBE = "fabricperf_throughput_flush"
LATENCY_RECV_PROBE = "recv_latency"
LATENCY_LEGACY_PTP_PARAM = "param_ptp_metrics"
LATENCY_MAP_NAME = "sr_latency"
LATENCY_LOCAL_EPOCH_PROBE = "fabricperf_latency_local_epoch"
LATENCY_LOCAL_SECOND_RECV_PROBE = "recv_latency_simple_b"
LATENCY_DEVFUNC_WAIT_ITERS = 1024
THROUGHPUT_MAP_NAME = "fabricperf_throughput"
MEMORY_MAP_NAME = "fabricperf_probe_bytes"


def load_probe_toml(probe_path: str | None) -> dict:
    """Load the original probe config using the same file/env convention as Neutrino."""
    if probe_path is not None:
        return toml.load(probe_path)
    probe_envvar = os.environ.get("NEUTRINO_PROBES")
    if probe_envvar is None:
        raise ValueError("Can not read probes from envaraible 'NEUTRINO_PROBES'")
    return toml.loads(probe_envvar)


def shell_probe_for(raw_probe: dict) -> dict:
    """Build a harmless entry-level probe preserving map and symbol metadata.

    Motivation: Neutrino core still needs to allocate result maps and initialize
    FabricPerf symbols, but the real trace snippets are inserted into the
    selected device function by this wrapper. Example: throughput keeps
    `fabricperf_throughput` map sizing while the entry receives only a no-op.
    """
    if raw_probe.get("dynamic") is True:
        raise ValueError("FabricPerf NCCL devFunc wrapper does not support dynamic maps yet")
    if "map" not in raw_probe:
        raise ValueError("FabricPerf NCCL devFunc wrapper requires probe maps")

    # Step: latency's PTP calibration is an entry-level synchronization block.
    # OLD: latency also received the no-op shell, which forced the full PTP
    # block into the selected devFunc fallback and could deadlock before NCCL
    # reached its SendRecv body.
    if is_latency_probe(raw_probe) and "ptp_metrics" in raw_probe.get("probe", {}):
        ptp_probe = copy.deepcopy(raw_probe["probe"]["ptp_metrics"])
        if "before" in ptp_probe:
            # Step: generated modules expose map bases through globals, not the
            # legacy map parameter name used by the original latency snippet.
            # Example: `param_ptp_metrics` becomes `__neutrino_map_ptr_0`.
            ptp_probe["before"] = rewrite_legacy_latency_param_load(ptp_probe["before"], raw_probe)
        shell = {
            "regs": max(1, int(raw_probe.get("regs", 1))),
            "probe": {
                "ptp_metrics": ptp_probe,
            },
            "map": raw_probe["map"],
        }
        if "symbol" in raw_probe:
            shell["symbol"] = raw_probe["symbol"]
        return shell

    shell = {
        "regs": max(1, int(raw_probe.get("regs", 1))),
        "probe": {
            "fabricperf_nccl_devfunc_shell": {
                "level": "thread",
                "pos": "kernel",
                "before": "{\n    mov.u64 %NR0, %NR0;\n}\n",
            }
        },
        "map": raw_probe["map"],
    }
    if "symbol" in raw_probe:
        shell["symbol"] = raw_probe["symbol"]
    return shell


def is_latency_probe(raw_probe: dict) -> bool:
    """Return whether one probe config is FabricPerf latency mode.

    Motivation: selected-devFunc latency needs a no-fabric runtime fallback, but
    throughput and memory must keep their original probe contracts. Example:
    `latency.probe` has `recv_latency` and `sr_latency`.
    """
    return (
        LATENCY_RECV_PROBE in raw_probe.get("probe", {})
        and LATENCY_MAP_NAME in raw_probe.get("map", {})
    )


def is_memory_probe(raw_probe: dict) -> bool:
    """Return whether one probe config is FabricPerf memory-byte mode.

    Motivation: memory mode can safely instrument every local NCCL devFunc
    candidate because its snippets count generic global-memory instructions.
    Example: AllGather may dispatch funcIds 0..5, and each body should count its
    own loads/stores without relying on a selected-only trap path.
    """
    return MEMORY_MAP_NAME in raw_probe.get("map", {})


def latency_local_arrival_snippet(prefix: str, done_label: str, step: int) -> str:
    """Emit a local receive-arrival timing record for selected devFunc latency.

    Motivation: CUDA fabric handles may be unavailable, so selected-devFunc
    latency needs a map-only fallback. Example: the value stored as `latency`
    is `%globaltimer - %NR0`, where `%NR0` is set at devFunc entry.
    """
    return f"""
{{
    .reg .pred %{prefix}_p<2>;
    .reg .b32 %{prefix}_r<2>;
    .reg .b64 %{prefix}_rd<5>;

    // Step: each active warp leader writes one sr_latency record per receive hook.
    mov.u32 %{prefix}_r1, %laneid;
    setp.ne.u32 %{prefix}_p1, %{prefix}_r1, 0;
    @%{prefix}_p1 bra {done_label};

    // Step: record local arrival timing relative to selected devFunc entry.
    mov.u64 %{prefix}_rd1, %globaltimer;
    sub.s64 %{prefix}_rd2, %{prefix}_rd1, %NR0;
    cvt.u64.u32 %{prefix}_rd3, %ctaid.x;
    mov.u64 %{prefix}_rd4, {step};
    SAVE [ sr_latency ] {{ %{prefix}_rd2, %{prefix}_rd3, %{prefix}_rd4 }};

{done_label}:
}}
""".strip()


def latency_local_devfunc_probe(raw_probe: dict) -> dict | None:
    """Return a selected-devFunc latency probe that does not use fabric buffers.

    Motivation: the full PTP latency probe requires `CU_MEM_HANDLE_TYPE_FABRIC`
    allocations, which can fail on systems without IMEX/fabric permissions.
    Example: this fallback still writes `sr_latency` rows for SendRecv receive
    arrivals while keeping FabricPerf/Neutrino map and reader behavior intact.
    """
    if not is_latency_probe(raw_probe):
        return None

    if LATENCY_MAP_NAME not in raw_probe.get("map", {}):
        return None

    # Preserve the public sr_latency schema while dropping PTP/fabric maps and symbols.
    local_probe = {
        "regs": max(1, int(raw_probe.get("regs", 0))),
        "symbol": {
            # Keep one rank-owned symbol so latency_analysis can infer rank from event.log.
            "deviceId": "u32",
        },
        "map": {
            LATENCY_MAP_NAME: copy.deepcopy(raw_probe["map"][LATENCY_MAP_NAME]),
        },
        "probe": {
            LATENCY_LOCAL_EPOCH_PROBE: {
                "level": "thread",
                "pos": "kernel",
                "before": "\n".join([
                    "{",
                    "    .reg .b32 %fplat_epoch_r<2>;",
                    "    // Step: touch deviceId so the rank symbol remains live.",
                    "    ld.const.u32 %fplat_epoch_r1, [deviceId];",
                    "    // Step: establish local selected-devFunc timing epoch.",
                    "    mov.u64 %NR0, %globaltimer;",
                    "}",
                ]),
            },
            LATENCY_RECV_PROBE: {
                "level": "thread",
                "match": {
                    "op": "ld.volatile.global.v2.b64",
                    "nth": 10,
                },
                "after": latency_local_arrival_snippet("fplat_recv", "$L__fplat_recv_done", 0),
            },
        },
    }
    return local_probe


def selected_devfunc_probe_config(raw_probe: dict) -> tuple[dict, str | None]:
    """Return the probe config FabricPerf should hand to selected-devFunc JIT.

    Motivation: selected NCCL devFunc latency cannot rely on the entry-kernel
    fixed-pair mailbox anchors: the SendRecv send-side hook is not active for
    every receive workgroup. Example: latency mode uses a local receive-arrival
    timing probe in the selected devFunc, while throughput and memory keep their
    original public contracts.
    """
    latency_local = latency_local_devfunc_probe(raw_probe)
    if latency_local is not None:
        return latency_local, "latency-local-devfunc"
    return raw_probe, None


def core_cuda_prober_path() -> Path:
    """Return the installed/source CUDA prober path without consulting env overrides."""
    return Path(cuda_probe.__file__).resolve()


def run_core_prober(workdir: Path, kernel_name: str, shell_probe_path: Path) -> int:
    """Run Neutrino's CUDA prober as the first stage."""
    python = os.environ.get("NEUTRINO_PYTHON", sys.executable)
    command = [
        python,
        str(core_cuda_prober_path()),
        str(workdir),
        kernel_name.encode("utf-8", "ignore").decode("utf-8", "ignore"),
        str(shell_probe_path),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    (workdir / "fabricperf_core_stdout.txt").write_text(result.stdout, encoding="utf-8")
    (workdir / "fabricperf_core_stderr.txt").write_text(result.stderr, encoding="utf-8")
    return result.returncode


def ptxas_options_from_log(process_log: Path) -> list[str]:
    """Recover preserved ptxas options from the first-stage process log."""
    if not process_log.exists():
        return []
    options: list[str] = []
    for match in PTXAS_OPTION_RE.finditer(process_log.read_text(encoding="utf-8", errors="replace")):
        option = match.group("option").replace(" ", "=")
        if option not in options:
            options.append(option)
    return options


def generated_decl_line(line: str, raw_probe: dict) -> bool:
    """Return whether a PTX declaration belongs to Neutrino/FabricPerf metadata.

    Motivation: selected devFunc instrumentation inserts map/symbol references
    into a `.func` that can appear before Neutrino's generated declarations.
    Example: memory mode needs `__neutrino_map_ptr_0` and `launchIndex`
    declared before `_Z20ncclDevFunc_SendRecvv`.
    """
    stripped = line.strip()
    if not stripped.endswith(";"):
        return False
    if not stripped.startswith("."):
        return False
    if PTX_GENERATED_SYMBOL_RE.search(stripped) is not None:
        return True
    for symbol_name in raw_probe.get("symbol", {}):
        if re.search(rf"\b{re.escape(symbol_name)}\b", stripped) is not None:
            return True
    return False


def hoist_generated_decls(ptx: str, raw_probe: dict) -> str:
    """Move generated map/symbol declarations before the first PTX function.

    Motivation: ptxas does not accept a pre-entry `.func` that references map
    globals or probe constants declared later in the module. Example: after
    probing `ncclDevFunc_SendRecv`, `ld.global ... [__neutrino_map_ptr_0]`
    must see the declaration before the function body.
    """
    lines = ptx.splitlines(keepends=True)
    hoisted: list[str] = []
    kept: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if generated_decl_line(line, raw_probe):
            key = line.strip()
            if key not in seen:
                hoisted.append(line)
                seen.add(key)
            continue
        kept.append(line)
    if len(hoisted) == 0:
        return ptx
    insert_at = 0
    for idx, line in enumerate(kept):
        if PTX_FUNCTION_OR_ENTRY_RE.match(line):
            insert_at = idx
            break
    else:
        return "".join(hoisted + kept)
    return "".join(kept[:insert_at] + hoisted + ["\n"] + kept[insert_at:])


def function_span(ptx: str, function_name: str) -> tuple[int, int, int, int, str]:
    """Return line/function/body span for one PTX `.func`.

    The tuple is `(line_start, open_brace, close_brace, line_end, header)`.
    """
    header_re = re.compile(
        rf"(?m)^(?P<header>\s*(?:\.visible\s+)?\.func\b[^\n]*\b{re.escape(function_name)}\s*\([^\n]*)"
    )
    match = header_re.search(ptx)
    if match is None:
        raise ValueError(f"{function_name} is not in a supported .func header")
    line_start = match.start()
    open_brace = ptx.find("{", match.end())
    if open_brace == -1:
        raise ValueError(f"{function_name} missing function body")
    close_brace = cuda_probe.find_matching_brace(ptx, open_brace)
    line_end = close_brace
    if line_end < len(ptx) and ptx[line_end:line_end + 1] == "\n":
        line_end += 1
    return line_start, open_brace, close_brace, line_end, ptx[line_start:open_brace]


def extract_body(ptx: str, open_brace: int) -> tuple[str, int]:
    """Return the body text and closing brace index for a PTX function/entry."""
    close_brace = cuda_probe.find_matching_brace(ptx, open_brace)
    return ptx[open_brace + 1:close_brace - 1], close_brace


def sendrecv_devfunc_loop_match() -> dict:
    """Return the compact SendRecv devFunc receive-loop anchor.

    Motivation: some NCCL SendRecv devFuncs lower receive progress as a data
    load followed by a later loop branch, so the entry-level receive signature
    can miss. Example: the selected `_Z20ncclDevFunc_SendRecvv` body has
    `ld.volatile.global.v4.u32` inside the copy loop before the outer branch.
    """
    return {
        "kind": "branch",
        "predicated": True,
        # OLD: ["st.volatile.global.v4.u32", "setp.lt.u64"] matched a store-side
        # path that the non-D SendRecv devFunc did not dynamically reach.
        "nearby_before_ops": ["ld.volatile.global.v4.u32", "setp.lt.u64"],
        "nearby_window": 160,
        "nth": 2,
    }


def devfunc_fallback_probe(raw_probe: dict, probe_name: str) -> dict | None:
    """Return a probe copy using the SendRecv devFunc receive-loop fallback."""
    if probe_name not in raw_probe.get("probe", {}):
        return None
    fallback = copy.deepcopy(raw_probe)
    fallback["probe"][probe_name]["match"] = sendrecv_devfunc_loop_match()
    return fallback


def map_symbol_index(raw_probe: dict, map_name: str) -> int:
    """Return the generated Neutrino map-symbol index for one map.

    Motivation: FabricPerf's selected-devFunc fallback may need to recompute a
    CTA-owned map base inside a non-entry function. Example: throughput's first
    map is available through `__neutrino_map_ptr_0`.
    """
    map_names = list(raw_probe.get("map", {}).keys())
    if map_name not in map_names:
        raise ValueError(f"FabricPerf probe is missing map {map_name!r}")
    return map_names.index(map_name)


def devfunc_throughput_map_base_snippet(raw_probe: dict) -> str:
    """Emit PTX that recomputes the first-warp map base for the current CTA.

    Motivation: Neutrino's generated `%map_*1` register is written only by
    `laneid == 0`. The SendRecv devFunc fallback selects NCCL's logical leader
    (`%r18 == 0`), which is not guaranteed to be warp lane zero. Example: this
    code stores into the same first-warp CTA stream the analyzer already reads.
    """
    map_index = map_symbol_index(raw_probe, THROUGHPUT_MAP_NAME)
    return f"""
    // Step: recompute the CTA first-warp map base inside the selected devFunc.
    // OLD: `mov.u64 ..., %map_fabricperf_throughput1` required the writer to
    // be Neutrino's warp leader, which NCCL's `%r18 == 0` leader need not be.
    mov.u32 %fpthr_recv_r15, %ntid.x;
    mov.u32 %fpthr_recv_r16, %ntid.y;
    mov.u32 %fpthr_recv_r17, %ntid.z;
    mul.lo.u32 %fpthr_recv_r18, %fpthr_recv_r15, %fpthr_recv_r16;
    mul.lo.u32 %fpthr_recv_r18, %fpthr_recv_r18, %fpthr_recv_r17;
    shr.u32 %fpthr_recv_r18, %fpthr_recv_r18, 5;
    mov.u32 %fpthr_recv_r19, %ctaid.x;
    mov.u32 %fpthr_recv_r20, %ctaid.y;
    mov.u32 %fpthr_recv_r21, %ctaid.z;
    mov.u32 %fpthr_recv_r22, %nctaid.x;
    mov.u32 %fpthr_recv_r23, %nctaid.y;
    mad.lo.s32 %fpthr_recv_r24, %fpthr_recv_r23, %fpthr_recv_r21, %fpthr_recv_r20;
    mad.lo.s32 %fpthr_recv_r25, %fpthr_recv_r24, %fpthr_recv_r22, %fpthr_recv_r19;
    mul.lo.u32 %fpthr_recv_r26, %fpthr_recv_r25, %fpthr_recv_r18;
    mul.wide.u32 %fpthr_recv_rd4, %fpthr_recv_r26, 16384;
    ld.global.u64 %fpthr_recv_rd3, [__neutrino_map_ptr_{map_index}];
    add.s64 %fpthr_recv_rd5, %fpthr_recv_rd3, %fpthr_recv_rd4;""".strip("\n")


def devfunc_throughput_flush_map_base_snippet(raw_probe: dict) -> str:
    """Emit PTX that recomputes the first-warp map base in the flush hook.

    Motivation: the same non-warp-leader writer that records arrivals also
    publishes headers at `ret`. Example: `%NR1 > 0` identifies the selected
    `%r18 == 0` writer, so the flush hook cannot rely on `%map_*1` either.
    """
    map_index = map_symbol_index(raw_probe, THROUGHPUT_MAP_NAME)
    return f"""
    // Step: recompute the CTA first-warp map base for the devFunc flush writer.
    // OLD: `mov.u64 ..., %map_fabricperf_throughput1` only worked for
    // Neutrino's warp leader, not necessarily NCCL's logical workgroup leader.
    mov.u32 %fpthr_flush_r4, %ntid.x;
    mov.u32 %fpthr_flush_r5, %ntid.y;
    mov.u32 %fpthr_flush_r6, %ntid.z;
    mul.lo.u32 %fpthr_flush_r7, %fpthr_flush_r4, %fpthr_flush_r5;
    mul.lo.u32 %fpthr_flush_r7, %fpthr_flush_r7, %fpthr_flush_r6;
    shr.u32 %fpthr_flush_r7, %fpthr_flush_r7, 5;
    mov.u32 %fpthr_flush_r8, %ctaid.x;
    mov.u32 %fpthr_flush_r9, %ctaid.y;
    mov.u32 %fpthr_flush_r10, %ctaid.z;
    mov.u32 %fpthr_flush_r11, %nctaid.x;
    mov.u32 %fpthr_flush_r12, %nctaid.y;
    mad.lo.s32 %fpthr_flush_r13, %fpthr_flush_r12, %fpthr_flush_r10, %fpthr_flush_r9;
    mad.lo.s32 %fpthr_flush_r14, %fpthr_flush_r13, %fpthr_flush_r11, %fpthr_flush_r8;
    mul.lo.u32 %fpthr_flush_r15, %fpthr_flush_r14, %fpthr_flush_r7;
    mul.wide.u32 %fpthr_flush_rd4, %fpthr_flush_r15, 16384;
    ld.global.u64 %fpthr_flush_rd3, [__neutrino_map_ptr_{map_index}];
    add.s64 %fpthr_flush_rd5, %fpthr_flush_rd3, %fpthr_flush_rd4;

    // Step: cell 0 is duration; cell 1 is total arrivals.""".strip("\n")


def apply_throughput_devfunc_capture_overrides(fallback: dict, raw_probe: dict) -> None:
    """Specialize throughput captures for `_Z20ncclDevFunc_SendRecvv`.

    Motivation: entry-kernel matching can usually recover a Neutrino workgroup,
    but the selected SendRecv devFunc fallback cannot. Example: `laneid == 0`
    keeps one writer per warp, and each such writer owns a valid Neutrino
    warp-level map slot that the analyzer can decode independently.
    """
    recv_probe = fallback["probe"][THROUGHPUT_RECV_PROBE]
    capture = recv_probe.setdefault("capture", {})
    # OLD: capture defaults used `%tid.x == 0`, which misses receive workgroups
    # when the active NCCL group does not include CTA thread zero.
    capture["group"] = {"value": "0"}
    # OLD: `%r18 == 0` was closer to NCCL's workgroup leader, but runtime
    # evidence showed no writer at the chosen post-loop anchor for SendRecv.
    capture["group_source"] = {"value": "%laneid"}
    capture["group_threads"] = {"value": "1"}
    # These captures are not used by the fixed-cell throughput snippet. The
    # devFunc fallback anchor is intentionally different from the entry-kernel
    # receive anchor, so keeping the generic capture rules makes matching fail.
    capture["recv_step"] = {"value": "0"}
    capture["recv_peer"] = {"value": "0"}
    capture["vblock"] = {"value": "%ctaid.x"}
    capture["chunk"] = {"value": "0"}


def namespace_throughput_recv_snippet(snippet: str, suffix: str) -> str:
    """Rename local PTX names in a copied throughput receive snippet.

    Motivation: the selected SendRecv devFunc Simple path has two active
    `ld.volatile.global.v2.b64` receive clusters. Example: the second hook must
    not reuse `$L__fpthr_recv_done` from the first hook in the same function.
    """
    return (
        snippet
        .replace("%fpthr_recv_", f"%fpthr_recv_{suffix}_")
        .replace("$L__fpthr_recv_done", f"$L__fpthr_recv_{suffix}_done")
    )


def throughput_devfunc_fallback_probe(raw_probe: dict) -> dict | None:
    """Return a devFunc-specific throughput probe fallback when applicable."""
    if THROUGHPUT_RECV_PROBE not in raw_probe.get("probe", {}):
        return None
    fallback = copy.deepcopy(raw_probe)
    recv_probe = fallback["probe"][THROUGHPUT_RECV_PROBE]
    # OLD: `sendrecv_devfunc_loop_match()` targeted an LL-style
    # `ld.volatile.global.v4.u32` poll region that the non-D SendRecv runtime
    # did not execute. Receive markers hit v2.b64 nth 10 and 26 instead.
    recv_probe["match"] = {
        "op": "ld.volatile.global.v2.b64",
        "nth": 10,
    }
    apply_throughput_devfunc_capture_overrides(fallback, raw_probe)
    second_probe = copy.deepcopy(fallback["probe"][THROUGHPUT_RECV_PROBE])
    second_probe["match"] = {
        "op": "ld.volatile.global.v2.b64",
        "nth": 26,
    }
    second_probe["after"] = namespace_throughput_recv_snippet(second_probe["after"], "b")
    fallback["probe"]["fabricperf_throughput_recv_simple_b"] = second_probe
    return fallback


def add_latency_devfunc_thread0_guard(snippet: str, prefix: str, done_label: str, pred_idx: int) -> str:
    """Gate a latency snippet to warp lane zero for selected devFunc mode.

    Motivation: Neutrino's generated map pointer is initialized by its warp map
    leader. Example: selected SendRecv latency should not let every lane
    execute `SAVE [ sr_latency ]` with an uninitialized `%map_sr_latency1`.
    """
    guard = "\n".join([
        f"    // Step: selected-devFunc latency uses one warp-lane writer.",
        f"    mov.u32 %{prefix}_r4, %laneid;",
        f"    setp.ne.u32 %{prefix}_p{pred_idx}, %{prefix}_r4, 0;",
        f"    @%{prefix}_p{pred_idx} bra {done_label};",
    ])
    register_anchor = f"    .reg .b64 %{prefix}_rd"
    anchor_idx = snippet.find(register_anchor)
    if anchor_idx == -1:
        raise ValueError(f"selected-devFunc latency guard could not find {prefix} register block")
    line_end = snippet.find("\n", anchor_idx)
    if line_end == -1:
        raise ValueError(f"selected-devFunc latency guard could not find {prefix} register line end")
    insert_at = line_end + 1
    return snippet[:insert_at] + guard + "\n" + snippet[insert_at:]


def bound_latency_devfunc_recv_wait(snippet: str) -> str:
    """Bound the selected-devFunc receive wait for peer timestamp messages.

    Motivation: the original latency probe can wait forever if the selected
    NCCL send-side hook does not run for the same workgroup. Example: SendRecv
    still progresses when a peer timestamp is missing, while valid messages are
    consumed and written to `sr_latency`.
    """
    old_wait = "\n".join([
        "$L__srlat_wait_msg:",
        "    ld.global.volatile.v2.u64 { %srlat_rd11, %srlat_rd12 }, [%srlat_rd9];",
        "    setp.ne.s64 %srlat_p3, %srlat_rd11, %srlat_rd10;",
        "    @%srlat_p3 bra $L__srlat_wait_msg;",
    ])
    new_wait = "\n".join([
        "    // Step: avoid deadlocking NCCL if the paired send hook did not publish.",
        "    mov.u32 %srlat_r4, 0;",
        "$L__srlat_wait_msg:",
        "    ld.global.volatile.v2.u64 { %srlat_rd11, %srlat_rd12 }, [%srlat_rd9];",
        "    setp.eq.s64 %srlat_p3, %srlat_rd11, %srlat_rd10;",
        "    @%srlat_p3 bra $L__srlat_have_msg;",
        "    add.u32 %srlat_r4, %srlat_r4, 1;",
        f"    setp.lt.u32 %srlat_p3, %srlat_r4, {LATENCY_DEVFUNC_WAIT_ITERS};",
        "    @%srlat_p3 bra $L__srlat_wait_msg;",
        "    bra.uni $L__srlat_done;",
        "$L__srlat_have_msg:",
    ])
    if old_wait not in snippet:
        # Already-bounded probes from latency.probe are valid inputs too, but
        # selected devFuncs must use the short bound to avoid stalling NCCL.
        # Example: the canonical latency.probe uses a long entry-kernel wait;
        # the copied SendRecv devFunc hook gets normalized to 1024 polls.
        bounded, replacements = re.subn(
            r"setp\.lt\.u32\s+%srlat_p(?P<pred>[34]),\s+%srlat_r(?P<reg>[45]),\s+\d+;",
            lambda match: (
                f"setp.lt.u32 %srlat_p{match.group('pred')}, "
                f"%srlat_r{match.group('reg')}, {LATENCY_DEVFUNC_WAIT_ITERS};"
            ),
            snippet,
            count=1,
        )
        if replacements > 0 and "$L__srlat_have_msg:" in snippet:
            return bounded
        raise ValueError("selected-devFunc latency wait block not found")
    return snippet.replace(old_wait, new_wait, 1)


def inject_latency_devfunc_recv_publish(snippet: str) -> str:
    """Publish a selected-devFunc receive timestamp to the peer slot.

    Motivation: NCCL SendRecv may not execute the separate send-side anchor for
    the same workgroup that reaches the receive hook. Example: both ranks can
    exchange receive-arrival timestamps by writing to the peer mailbox buffer
    immediately before the bounded wait.
    """
    if "publish this receive-arrival timestamp to the peer mailbox" in snippet:
        return snippet

    reg_match = re.search(r"(?m)^    \.reg \.b64 %srlat_rd<(?P<count>\d+)>;", snippet)
    if reg_match is None:
        raise ValueError("selected-devFunc latency register block not found")

    tag_anchor = "    or.b64 %srlat_rd11, %srlat_rd11, 160;"
    if tag_anchor in snippet:
        if int(reg_match.group("count")) < 28:
            snippet = (
                snippet[:reg_match.start()]
                + "    .reg .b64 %srlat_rd<28>;"
                + snippet[reg_match.end():]
            )
        publish = "\n".join([
            tag_anchor,
            "    // Step: publish this receive-arrival timestamp to the peer mailbox.",
            "    mul.wide.u32 %srlat_rd22, %srlat_r3, 8;",
            "    add.s64 %srlat_rd22, %srlat_rd2, %srlat_rd22;",
            "    ld.global.u64 %srlat_rd23, [%srlat_rd22];",
            "    mul.wide.u32 %srlat_rd25, %srlat_r1, 4096;",
            "    cvt.u64.u32 %srlat_rd26, %srlat_r4;",
            "    add.s64 %srlat_rd25, %srlat_rd25, %srlat_rd26;",
            "    mul.lo.u64 %srlat_rd25, %srlat_rd25, 16;",
            "    add.s64 %srlat_rd26, %srlat_rd23, %srlat_rd25;",
            "    add.s64 %srlat_rd27, %srlat_rd26, 8;",
            "    st.global.volatile.u64 [%srlat_rd27], %srlat_rd1;",
            "    fence.sc.sys;",
            "    st.global.volatile.u64 [%srlat_rd26], %srlat_rd11;",
            "    fence.sc.sys;",
        ])
        return snippet.replace(tag_anchor, publish, 1)

    reg_line = "    .reg .b64 %srlat_rd<22>;"
    if reg_line not in snippet:
        raise ValueError("selected-devFunc latency register block not found")
    snippet = snippet.replace(reg_line, "    .reg .b64 %srlat_rd<24>;", 1)

    anchor = "    add.s64 %srlat_rd9, %srlat_rd5, %srlat_rd8;"
    publish = "\n".join([
        anchor,
        "    // Step: publish this receive-arrival timestamp to the peer slot.",
        "    mul.wide.u32 %srlat_rd21, %srlat_r3, 8;",
        "    add.s64 %srlat_rd22, %srlat_rd2, %srlat_rd21;",
        "    ld.global.u64 %srlat_rd23, [%srlat_rd22];",
        "    add.s64 %srlat_rd23, %srlat_rd23, %srlat_rd8;",
        "    st.global.volatile.v2.u64 [%srlat_rd23], { %srlat_rd10, %srlat_rd1 };",
        "    fence.sc.sys;",
    ])
    if anchor not in snippet:
        raise ValueError("selected-devFunc latency receive address anchor not found")
    return snippet.replace(anchor, publish, 1)


def latency_devfunc_fallback_probe(raw_probe: dict) -> dict | None:
    """Return a devFunc-specific latency probe fallback when applicable."""
    if LATENCY_RECV_PROBE not in raw_probe.get("probe", {}):
        return None
    fallback = copy.deepcopy(raw_probe)
    # Step: keep latency's PTP calibration at the real kernel entry. The
    # selected devFunc body only gets SendRecv message hooks. Example:
    # `ptp_metrics` still runs through `shell_probe_for`, while this fallback
    # writes `sr_latency` from inside `_Z20ncclDevFunc_SendRecvv`.
    fallback["probe"] = {
        name: fallback["probe"][name]
        for name in (LATENCY_RECV_PROBE, "send_timestamp_message")
        if name in fallback["probe"]
    }
    recv_probe = fallback["probe"][LATENCY_RECV_PROBE]
    # OLD: the LL-style branch fallback assembled in some static experiments
    # but runtime markers showed the non-D SendRecv smoke uses Simple v2.b64
    # receive clusters instead.
    recv_probe["match"] = {
        "op": "ld.volatile.global.v2.b64",
        "nth": 10,
    }
    recv_capture = recv_probe.setdefault("capture", {})
    recv_capture["recv_step"] = {"value": "0"}
    recv_capture["recv_peer"] = {"value": "%srlat_r3"}
    # Step: pair send/receive messages by NCCL's virtual workgroup when the
    # PTX pattern exposes it. OLD: `%ctaid.x` can mismatch SendRecv workgroups.
    recv_capture["vblock"] = {"from": "virtual_block"}
    recv_capture["chunk"] = {"value": "0"}
    # Keep the send-side timestamp as the only mailbox publisher. A receive-side
    # timestamp uses a different event time and can overwrite the matching send
    # tag before the peer consumes it, producing cross-clock artifacts.
    recv_probe["after"] = bound_latency_devfunc_recv_wait(recv_probe["after"])
    recv_probe["after"] = add_latency_devfunc_thread0_guard(
        recv_probe["after"],
        "srlat",
        "$L__srlat_done",
        4,
    )
    if "send_timestamp_message" in fallback.get("probe", {}):
        send_probe = fallback["probe"]["send_timestamp_message"]
        # Step: send a timestamp before the selected receive wait. OLD:
        # after_ref could deadlock once the receive hook became active.
        send_probe["match"] = {
            "op": "ld.volatile.global.u64",
            "before_ref": LATENCY_RECV_PROBE,
            "last": True,
        }
        send_capture = send_probe.setdefault("capture", {})
        send_capture["send_step"] = {"value": "0"}
        send_capture["send_peer"] = {"value": "%srmsg_r3"}
        send_probe["after"] = add_latency_devfunc_thread0_guard(
            send_probe["after"],
            "srmsg",
            "$L__srmsg_done",
            3,
        )
    return fallback


def probe_function_body_once(function_body: str, raw_probe: dict) -> str:
    """Instrument one no-argument device function body with Neutrino snippets.

    Motivation: `cuda.py` instruments entries, so FabricPerf wraps the selected
    devFunc body in a temporary entry, reuses Neutrino's snippet machinery, and
    then unwraps the instrumented body.
    """
    probes, maps, regs, _ = load(raw_probe, include_symbols=True)
    fake_entry = ".visible .entry __fabricperf_nccl_devfunc_proxy()\n{\n" + function_body + "\n}\n"
    probed_fake = cuda_probe.probing(fake_entry, probes, maps, regs)
    open_brace = probed_fake.find("{")
    if open_brace == -1:
        raise ValueError("internal proxy entry missing body")
    body, _ = extract_body(probed_fake, open_brace)
    return body


def is_latency_local_devfunc_probe(raw_probe: dict) -> bool:
    """Return whether a latency probe is already specialized to local devFunc timing."""
    return LATENCY_LOCAL_EPOCH_PROBE in raw_probe.get("probe", {})


def count_ptx_op(function_body: str, op: str) -> int:
    """Return the number of PTX instructions with a matching opcode."""
    op_re = re.compile(rf"(?m)^\s*(?:@\S+\s+)?{re.escape(op)}\b")
    return len(op_re.findall(function_body))


def expand_latency_local_devfunc_probe(function_body: str, raw_probe: dict) -> dict:
    """Clone local latency receive probes for every matching SendRecv load.

    Motivation: the active Simple-protocol receive poll ordinal changes across
    NCCL builds. Example: one sm_90 build exposes 22 `ld.volatile.global.v2.b64`
    instances, so a fixed `nth=10` can assemble but never execute.
    """
    recv_count = count_ptx_op(function_body, "ld.volatile.global.v2.b64")
    if recv_count <= 0:
        return raw_probe

    expanded = copy.deepcopy(raw_probe)
    probes = {
        name: probe
        for name, probe in expanded.get("probe", {}).items()
        if name not in (LATENCY_RECV_PROBE, LATENCY_LOCAL_SECOND_RECV_PROBE)
    }
    for nth in range(1, recv_count + 1):
        name = LATENCY_RECV_PROBE if nth == 1 else f"{LATENCY_RECV_PROBE}_local_{nth}"
        probes[name] = {
            "level": "thread",
            "match": {
                "op": "ld.volatile.global.v2.b64",
                "nth": nth,
            },
            "after": latency_local_arrival_snippet(
                f"fplat_recv_{nth}",
                f"L__fplat_recv_{nth}_done",
                nth - 1,
            ),
        }
    expanded["probe"] = probes
    return expanded


def probe_function_body(function_body: str, raw_probe: dict) -> tuple[str, str | None]:
    """Instrument a device function body, retrying known devFunc fallbacks."""
    if is_latency_probe(raw_probe):
        if is_latency_local_devfunc_probe(raw_probe):
            expanded = expand_latency_local_devfunc_probe(function_body, raw_probe)
            return probe_function_body_once(function_body, expanded), None
        fallback = latency_devfunc_fallback_probe(raw_probe)
        if fallback is not None:
            return probe_function_body_once(function_body, fallback), "latency-sendrecv-devfunc"

    try:
        return probe_function_body_once(function_body, raw_probe), None
    except ValueError as exc:
        if "found no matches" not in str(exc):
            raise
        fallbacks = [
            ("throughput-sendrecv-devfunc", throughput_devfunc_fallback_probe(raw_probe)),
            ("latency-sendrecv-devfunc", latency_devfunc_fallback_probe(raw_probe)),
        ]
        for fallback_name, fallback in fallbacks:
            if fallback is None:
                continue
            return probe_function_body_once(function_body, fallback), fallback_name
        raise


def rewrite_legacy_latency_param_load(function_body: str, raw_probe: dict) -> str:
    """Rewrite old latency map-parameter loads to generated map globals.

    Motivation: selected NCCL devFuncs do not have Neutrino's entry parameters.
    Example: latency's PTP snippet historically reads `param_ptp_metrics`; in
    a generated selected-devFunc module that base pointer lives in
    `__neutrino_map_ptr_0`.
    """
    if LATENCY_LEGACY_PTP_PARAM not in function_body:
        return function_body
    map_names = list(raw_probe.get("map", {}).keys())
    if "ptp_metrics" not in map_names:
        return function_body
    map_index = map_names.index("ptp_metrics")

    def replacement(match: re.Match[str]) -> str:
        """Preserve whether the legacy load was at text start or mid-body."""
        return f"{match.group('prefix')}    ld.global.u64 %ptp_rd76, [__neutrino_map_ptr_{map_index}];"

    return re.sub(
        r"(?P<prefix>\n|^)\s*ld\.param\.u64\s+%ptp_rd76,\s+\[param_ptp_metrics\];\n\s*cvta\.to\.global\.u64\s+%ptp_rd76,\s+%ptp_rd76;",
        replacement,
        function_body,
        count=1,
    )


def instrument_selected_devfunc(ptx: str, function_name: str, raw_probe: dict) -> tuple[str, str | None]:
    """Replace one local NCCL devFunc body with an instrumented body."""
    line_start, open_brace, close_brace, line_end, header = function_span(ptx, function_name)
    if ".param" in header:
        raise ValueError(f"{function_name} has parameters; FabricPerf wrapper only supports no-arg devFuncs")
    function_body = ptx[open_brace + 1:close_brace - 1]
    if re.search(r"(?m)^\s*\.reg\s+\.u64\s+%NR<", function_body) is not None:
        # Step: keep postprocess idempotent for copied/debug workdirs. OLD:
        # rerunning instrumentation duplicated `%NR`, `%buf`, and map registers.
        return ptx, None
    instrumented_body, fallback_name = probe_function_body(function_body, raw_probe)
    instrumented_body = rewrite_legacy_latency_param_load(instrumented_body, raw_probe)
    rewritten_function = ptx[line_start:open_brace + 1] + "\n" + instrumented_body + "\n}"
    if line_end > close_brace:
        rewritten_function += "\n"
    return ptx[:line_start] + rewritten_function + ptx[line_end:], fallback_name


def instrument_devfunc_candidates(
    ptx: str,
    target: nccl_devfunc_table.NcclDevFuncTarget,
    local_mapping: dict[int, str],
    raw_probe: dict,
) -> tuple[str, str | None, list[str], list[dict[str, str]]]:
    """Instrument the selected target and any safe same-module candidates.

    Motivation: memory mode needs all local NCCL protocol variants to be valid
    dispatch targets after the fallback becomes multi-way. Example: if AllGather
    runs with funcId 4, the local LL128 body should be callable and instrumented
    instead of trapping in a funcId-3-only fallback.
    """
    if is_memory_probe(raw_probe):
        # Stage: memory snippets are generic byte counters, so every local mapped
        # no-arg devFunc can collect meaningful load/store/cp.async bytes.
        candidate_names = [name for _, name in sorted(local_mapping.items())]
        if target.function_name not in candidate_names:
            candidate_names.append(target.function_name)
    else:
        # Stage: throughput/latency probes use protocol-specific anchors and
        # fallbacks; keep the historical selected-target scope for those modes.
        candidate_names = [target.function_name]

    fallback_name: str | None = None
    instrumented: list[str] = []
    skipped: list[dict[str, str]] = []
    seen: set[str] = set()
    for function_name in candidate_names:
        if function_name in seen:
            continue
        seen.add(function_name)
        try:
            ptx, candidate_fallback = instrument_selected_devfunc(ptx, function_name, raw_probe)
        except ValueError as exc:
            if function_name == target.function_name:
                raise
            skipped.append({"function": function_name, "reason": str(exc)})
            continue
        instrumented.append(function_name)
        if candidate_fallback is not None and fallback_name is None:
            fallback_name = candidate_fallback

    return ptx, fallback_name, instrumented, skipped


def nccl_direct_call_block(function_name: str) -> list[str]:
    """Emit PTX that directly calls one local no-argument NCCL devFunc."""
    return [
        "{ //",
        ".reg .b32 temp_param_reg;",
        "call.uni ",
        f"{function_name}, ",
        "(",
        ");",
        "} //",
    ]


def nccl_runtime_table_mapping_from_kernel_info(kernel_info_path: Path) -> dict[int, str]:
    """Parse generated NCCL runtime table metadata from `kernel.info`.

    Motivation: the core CUDA prober already records every local funcId-to-devFunc
    candidate in `NCCL_RUNTIME_TABLE`. Example: AllGather maps slots 0..5 to six
    local protocol functions, while the selected descriptor target is only slot 3.
    """
    if not kernel_info_path.exists():
        return {}
    lines = [
        line.strip()
        for line in kernel_info_path.read_text(encoding="utf-8", errors="replace").splitlines()
    ]
    try:
        start = lines.index(cuda_probe.NCCL_RUNTIME_TABLE_SECTION)
    except ValueError:
        return {}

    header_fields = 6
    if start + header_fields >= len(lines):
        return {}
    table_name = lines[start + 2]
    if table_name != "ncclDevFuncTable":
        return {}
    try:
        table_count = int(lines[start + 5])
        n_entries = int(lines[start + 6])
    except ValueError:
        return {}

    mapping: dict[int, str] = {}
    for line in lines[start + 7:start + 7 + n_entries]:
        if "," not in line:
            continue
        index_text, function_name = line.split(",", 1)
        try:
            table_index = int(index_text)
        except ValueError:
            continue
        if 0 <= table_index < table_count and function_name:
            mapping[table_index] = function_name
    return dict(sorted(mapping.items()))


def selected_target_func_id(
    target: nccl_devfunc_table.NcclDevFuncTarget,
    local_mapping: dict[int, str],
) -> int | None:
    """Return the funcId for the selected devFunc, falling back to kernel.info.

    Motivation: some NCCL wrappers expose many unrelated `setp.*.s16`
    comparisons, so descriptor inference can return None even though
    `kernel.info` has the exact local runtime-table mapping. Example: SendRecv
    maps slot 669 to `_Z20ncclDevFunc_SendRecvv`.
    """
    if target.function_id is not None:
        return target.function_id
    matches = [
        func_id
        for func_id, function_name in local_mapping.items()
        if function_name == target.function_name
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def rewrite_table_fallback_to_local_dispatch(ptx: str, local_mapping: dict[int, str]) -> tuple[str, int]:
    """Rewrite fallback table calls to a multi-way local direct-call dispatch.

    Motivation: selected-only fallback replacement traps valid runtime funcIds
    that select another local NCCL protocol body. Example: AllGather funcId 4
    should call the local LL128 function instead of hitting FabricPerf's old
    funcId-3-only `trap`.
    """
    if len(local_mapping) == 0:
        return ptx, 0
    return cuda_probe.rewrite_nccl_dev_func_table_fallback(ptx, local_mapping)


def find_indirect_call_end(lines: list[str], start_idx: int, call_reg: str) -> int | None:
    """Find the end of a cuobjdump multi-line indirect call block."""
    saw_call = False
    saw_target = False
    for idx in range(start_idx, min(len(lines), start_idx + 32)):
        stripped = lines[idx].strip()
        if stripped.startswith("call"):
            saw_call = True
        if saw_call and stripped.rstrip(",") == call_reg:
            saw_target = True
        if saw_target and stripped.startswith("}"):
            return idx
    return None


def rewrite_table_fallback_to_direct_call(ptx: str, function_id: int | None, function_name: str) -> tuple[str, int]:
    """Rewrite matching `ncclDevFuncTable` fallback calls to one direct call.

    This is the static replacement path. It avoids adding a patch kernel launch;
    the generated module calls the local instrumented devFunc body directly.
    """
    if function_id is None or "ncclDevFuncTable" not in ptx:
        return ptx, 0
    lines = ptx.splitlines()
    out: list[str] = []
    idx = 0
    replacements = 0
    while idx < len(lines):
        cvt = NCCL_TABLE_CVT_RE.match(lines[idx])
        if cvt is None or idx + 4 >= len(lines):
            out.append(lines[idx])
            idx += 1
            continue
        mul = NCCL_TABLE_MUL_RE.match(lines[idx + 1])
        mov = NCCL_TABLE_MOV_RE.match(lines[idx + 2])
        add = NCCL_TABLE_ADD_RE.match(lines[idx + 3])
        load = NCCL_TABLE_LD_RE.match(lines[idx + 4])
        if (
            mul is None
            or mov is None
            or add is None
            or load is None
            or mul.group("src") != cvt.group("dst")
            or add.group("table") != mov.group("dst")
            or add.group("offset") != mul.group("dst")
            or load.group("addr") != add.group("dst")
        ):
            out.append(lines[idx])
            idx += 1
            continue
        call_end = find_indirect_call_end(lines, idx + 5, load.group("dst"))
        if call_end is None:
            out.append(lines[idx])
            idx += 1
            continue
        pred = f"%fabricperf_ndft_p{replacements}"
        done = f"$L__fabricperf_nccl_devfunc_done_{replacements}"
        call = f"$L__fabricperf_nccl_devfunc_call_{replacements}_{function_id}"
        out.extend([
            "// begin FabricPerf static ncclDevFuncTable replacement",
            "{",
            f".reg .pred {pred};",
            f"setp.eq.s16 {pred}, {cvt.group('src')}, {function_id};",
            f"@{pred} bra {call};",
            "trap;",
            f"bra.uni {done};",
            f"{call}:",
        ])
        out.extend(nccl_direct_call_block(function_name))
        out.extend([
            f"bra.uni {done};",
            f"{done}:",
            "}",
            "// end FabricPerf static ncclDevFuncTable replacement",
        ])
        replacements += 1
        idx = call_end + 1
    return "\n".join(out) + "\n", replacements


def redirect_direct_dispatch_to_fallback(ptx: str, function_id: int | None) -> tuple[str, int]:
    """Route a known NCCL direct fast path through the fallback call block.

    Motivation: SendRecv has both an inlined direct funcId path and a table
    fallback. Example: older dumps used `setp.eq.s16` plus a direct branch,
    while NCCL 2.28.9 can use `setp.ne.s16` where the predicated branch already
    targets the fallback. In both shapes, FabricPerf forces the selected funcId
    through the rewritten fallback so it calls the instrumented local devFunc.
    """
    if function_id is None:
        return ptx, 0
    lines = ptx.splitlines()
    out = list(lines)
    redirects = 0
    for idx, line in enumerate(lines):
        dispatch = NCCL_DIRECT_DISPATCH_SET_RE.match(line)
        if dispatch is None or int(dispatch.group("func_id")) != function_id:
            continue
        pred = dispatch.group("pred")
        if dispatch.group("op") == "eq":
            for branch_idx in range(idx + 1, min(idx + 80, len(lines) - 1)):
                pred_branch = NCCL_PREDICATED_BRA_RE.match(lines[branch_idx])
                fallback_branch = NCCL_UNCONDITIONAL_BRA_RE.match(lines[branch_idx + 1])
                if pred_branch is None or fallback_branch is None or pred_branch.group("pred") != pred:
                    continue
                fallback_target = fallback_branch.group("target")
                if pred_branch.group("target") == fallback_target:
                    continue
                out[branch_idx] = "\n".join([
                    f"{pred_branch.group('indent')}// OLD: {lines[branch_idx].strip()}",
                    f"{pred_branch.group('indent')}@{pred} bra {fallback_target};",
                ])
                redirects += 1
                break
            continue

        for branch_idx in range(idx + 1, min(idx + 80, len(lines))):
            pred_branch = NCCL_PREDICATED_BRA_RE.match(lines[branch_idx])
            if pred_branch is None or pred_branch.group("pred") != pred:
                continue
            fallback_target = pred_branch.group("target")
            out[branch_idx] = "\n".join([
                f"{pred_branch.group('indent')}// OLD: {lines[branch_idx].strip()}",
                f"{pred_branch.group('indent')}bra.uni {fallback_target};",
            ])
            redirects += 1
            break
    return "\n".join(out) + "\n", redirects


def append_log(workdir: Path, message: str) -> None:
    """Append one FabricPerf wrapper diagnostic to the workdir process log."""
    with (workdir / "process.log").open("a", encoding="utf-8") as handle:
        print(f"[fabricperf-devfunc] {message}", file=handle)


def postprocess_generated_module(
    workdir: Path,
    kernel_name: str,
    raw_probe: dict,
    probe_config_label: str | None = None,
) -> dict:
    """Instrument selected devFunc PTX and reassemble generated binaries."""
    probed_path = workdir / "probed.ptx"
    pruned_path = workdir / "pruned.ptx"
    if not probed_path.exists():
        raise ValueError(f"{probed_path} not found")
    probed_ptx = probed_path.read_text(encoding="utf-8", errors="replace")
    _, _, entry_ptx, _ = cuda_probe.prune(probed_ptx, kernel_name)
    target = nccl_devfunc_table.select_target(kernel_name, probed_ptx, entry_ptx)
    metadata = {
        "kernel_name": kernel_name,
        "enabled": True,
        "status": "no-target",
        "target_function": None,
        "target_func_id": None,
        "target_source": None,
        "table_rewrites": 0,
        "direct_dispatch_redirects": 0,
        "probe_fallback": None,
        "candidates": [],
        "local_dispatch": {},
        "instrumented_functions": [],
        "skipped_functions": [],
    }
    if target is None:
        (workdir / "fabricperf_nccl_devfunc.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        append_log(workdir, "no matching local NCCL devFunc target")
        return metadata

    local_mapping = nccl_runtime_table_mapping_from_kernel_info(workdir / "kernel.info")
    if len(local_mapping) == 0 and target.function_id is not None:
        # OLD: the selected-only rewrite built this implicit one-entry mapping and
        # trapped every other funcId. Keep it only as a compatibility fallback when
        # older kernel.info files do not include NCCL_RUNTIME_TABLE metadata.
        local_mapping = {target.function_id: target.function_name}

    target_func_id = selected_target_func_id(target, local_mapping)

    metadata.update({
        "status": "instrumented",
        "target_function": target.function_name,
        "target_func_id": target_func_id,
        "target_source": target.source,
        "candidates": list(target.candidates),
        "local_dispatch": {str(key): value for key, value in sorted(local_mapping.items())},
    })
    append_log(workdir, f"target {target.function_name} func_id={target.function_id}")
    append_log(workdir, f"local dispatch entries={len(local_mapping)}")

    # OLD: probed_ptx, fallback_name = instrument_selected_devfunc(probed_ptx, target.function_name, raw_probe)
    probed_ptx, fallback_name, instrumented_functions, skipped_functions = instrument_devfunc_candidates(
        probed_ptx,
        target,
        local_mapping,
        raw_probe,
    )
    metadata["probe_fallback"] = fallback_name or probe_config_label
    metadata["instrumented_functions"] = instrumented_functions
    metadata["skipped_functions"] = skipped_functions
    if metadata["probe_fallback"] is not None:
        append_log(workdir, f"probe fallback {metadata['probe_fallback']}")
    for skipped in skipped_functions:
        append_log(workdir, f"skip candidate {skipped['function']}: {skipped['reason']}")
    probed_ptx, redirects = redirect_direct_dispatch_to_fallback(probed_ptx, target_func_id)
    # OLD: probed_ptx, rewrites = rewrite_table_fallback_to_direct_call(probed_ptx, target.function_id, target.function_name)
    probed_ptx, rewrites = rewrite_table_fallback_to_local_dispatch(probed_ptx, local_mapping)
    probed_ptx = hoist_generated_decls(probed_ptx, raw_probe)
    probed_ptx = cuda_probe.strip_source_annotations(probed_ptx)
    metadata["direct_dispatch_redirects"] = redirects
    metadata["table_rewrites"] = rewrites
    probed_path.write_text(probed_ptx, encoding="utf-8")
    (workdir / "fabricperf_nccl_devfunc.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if pruned_path.exists():
        pruned_ptx = pruned_path.read_text(encoding="utf-8", errors="replace")
        pruned_ptx, _ = redirect_direct_dispatch_to_fallback(pruned_ptx, target_func_id)
        # OLD: pruned_ptx, _ = rewrite_table_fallback_to_direct_call(pruned_ptx, target.function_id, target.function_name)
        pruned_ptx, _ = rewrite_table_fallback_to_local_dispatch(pruned_ptx, local_mapping)
        pruned_ptx = hoist_generated_decls(pruned_ptx, raw_probe)
        pruned_ptx = cuda_probe.strip_source_annotations(pruned_ptx)
        pruned_path.write_text(pruned_ptx, encoding="utf-8")

    options = ptxas_options_from_log(workdir / "process.log")
    with (workdir / "process.log").open("a", encoding="utf-8") as handle:
        old_log = cuda_probe.log
        cuda_probe.log = handle
        try:
            cuda_probe.assemble(str(workdir), "probed", options)
            if pruned_path.exists():
                cuda_probe.assemble(str(workdir), "pruned", options)
        finally:
            cuda_probe.log = old_log

    metadata["status"] = "assembled"
    (workdir / "fabricperf_nccl_devfunc.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    """Run the FabricPerf devFunc wrapper using Neutrino's prober contract."""
    if len(sys.argv) < 3:
        print("usage: nccl/prober.py <workdir> <kernel_name> [probe.toml]", file=sys.stderr)
        return 2

    workdir = Path(sys.argv[1])
    kernel_name = sys.argv[2].encode("utf-8", "ignore").decode("utf-8", "ignore")
    probe_path = sys.argv[3] if len(sys.argv) > 3 else None

    try:
        raw_probe = load_probe_toml(probe_path)
        active_probe, probe_config_label = selected_devfunc_probe_config(raw_probe)
        shell_probe_path = workdir / "fabricperf_nccl_devfunc_shell.probe"
        with shell_probe_path.open("w", encoding="utf-8") as handle:
            toml.dump(shell_probe_for(active_probe), handle)
        core_status = run_core_prober(workdir, kernel_name, shell_probe_path)
        if core_status != 0:
            append_log(workdir, f"core cuda prober failed status={core_status}")
            return core_status
        postprocess_generated_module(workdir, kernel_name, active_probe, probe_config_label)
        return 0
    except Exception as exc:
        metadata_path = workdir / "fabricperf_nccl_devfunc.json"
        metadata = {
            "kernel_name": kernel_name,
            "enabled": True,
        }
        if metadata_path.exists():
            try:
                metadata.update(json.loads(metadata_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                metadata["previous_metadata_invalid"] = True
        metadata["status"] = "failed"
        metadata["error"] = str(exc)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        append_log(workdir, f"failed: {exc}")
        print(f"[fabricperf-devfunc] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
