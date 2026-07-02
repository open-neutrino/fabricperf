#!/usr/bin/env bash
# Common helpers for FabricPerf nccl-tests benchmark smoke scripts.

set -euo pipefail

FABRICPERF_BENCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FABRICPERF_PLUGIN_DIR="$(cd -- "${FABRICPERF_BENCH_DIR}/.." && pwd)"
FABRICPERF_REPO_ROOT="$(cd -- "${FABRICPERF_PLUGIN_DIR}/../../.." && pwd)"
FABRICPERF_WORKSPACE_ROOT="$(cd -- "${FABRICPERF_REPO_ROOT}/.." && pwd)"
# Store transient files on the workspace filesystem unless the caller already
# selected a temp root. Example: TMPDIR="${FABRICPERF_WORKSPACE_ROOT}/fabricperf_tmp/tmp".
# OLD: bench runs inherited an unset TMPDIR and many tools defaulted to /tmp.
: "${TMPDIR:=${FABRICPERF_WORKSPACE_ROOT}/fabricperf_tmp/tmp}"
export TMPDIR
mkdir -p "${TMPDIR}"

fabricperf_die() {
    echo "[fabricperf-bench][error] $*" >&2
    exit 1
}

fabricperf_prepend_path() {
    local var_name="$1"
    local value="$2"
    [[ -d "${value}" ]] || return 0
    case ":${!var_name:-}:" in
        *":${value}:"*) ;;
        *) export "${var_name}=${value}${!var_name:+:${!var_name}}" ;;
    esac
}

fabricperf_detect_site_packages() {
    local env_dir
    for env_dir in \
        "${FABRICPERF_WORKSPACE_ROOT}/.venv-deepep" \
        "${FABRICPERF_WORKSPACE_ROOT}/.venv" \
        "${FABRICPERF_REPO_ROOT}/.venv"; do
        if [[ -d "${env_dir}" ]]; then
            find "${env_dir}" -path '*/site-packages' -type d -print -quit 2>/dev/null
            return 0
        fi
    done
}

fabricperf_detect_nccl_lib_dir() {
    local candidate
    local -a candidates=()
    [[ -n "${NCCL_LIB_DIR:-}" ]] && candidates+=("${NCCL_LIB_DIR}")
    if [[ -n "${NCCL_HOME:-}" ]]; then
        candidates+=("${NCCL_HOME}/lib" "${NCCL_HOME}/build/lib")
    fi
    candidates+=(
        "${FABRICPERF_WORKSPACE_ROOT}/nccl/build/lib"
        "${FABRICPERF_REPO_ROOT}/../nccl/build/lib"
    )
    for candidate in "${candidates[@]}"; do
        if [[ -n "${candidate}" && -d "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    candidate="$(find "${FABRICPERF_WORKSPACE_ROOT}" -path '*/nccl/build/lib/libnccl.so*' -type f -print -quit 2>/dev/null || true)"
    [[ -n "${candidate}" ]] && dirname "${candidate}"
}

fabricperf_setup_paths() {
    local site_packages
    site_packages="$(fabricperf_detect_site_packages || true)"
    [[ -n "${site_packages}" ]] && fabricperf_prepend_path PYTHONPATH "${site_packages}"
    fabricperf_prepend_path PYTHONPATH "${FABRICPERF_REPO_ROOT}"

    local nccl_lib_dir
    nccl_lib_dir="$(fabricperf_detect_nccl_lib_dir || true)"
    [[ -n "${nccl_lib_dir}" ]] && fabricperf_prepend_path LD_LIBRARY_PATH "${nccl_lib_dir}"
}

fabricperf_find_deepep_root() {
    local candidate
    for candidate in \
        "${DEEPEP_ROOT:-}" \
        "${FABRICPERF_WORKSPACE_ROOT}/DeepEP" \
        "${FABRICPERF_REPO_ROOT}/../DeepEP"; do
        if [[ -n "${candidate}" && -f "${candidate}/tests/elastic/test_ep.py" ]]; then
            echo "$(cd -- "${candidate}" && pwd)"
            return 0
        fi
    done
    fabricperf_die "could not find DeepEP checkout; set DEEPEP_ROOT"
}

fabricperf_find_deepep_python() {
    local candidate
    for candidate in \
        "${DEEPEP_PYTHON:-}" \
        "${FABRICPERF_WORKSPACE_ROOT}/.venv-deepep/bin/python" \
        "${FABRICPERF_WORKSPACE_ROOT}/.venv/bin/python" \
        "$(command -v python3 || true)"; do
        if [[ -n "${candidate}" && -x "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    fabricperf_die "could not find a Python executable for DeepEP; set DEEPEP_PYTHON"
}

fabricperf_detect_gpus() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
        return 0
    fi
    echo 0
}

fabricperf_default_np() {
    local preferred="${1:-2}"
    local detected
    detected="$(fabricperf_detect_gpus)"
    if [[ "${detected}" =~ ^[0-9]+$ && "${detected}" -gt 0 && "${detected}" -lt "${preferred}" ]]; then
        echo "${detected}"
        return 0
    fi
    echo "${preferred}"
}

fabricperf_scale_values() {
    if [[ -n "${FABRICPERF_BENCH_GPU_COUNTS:-}" ]]; then
        echo "${FABRICPERF_BENCH_GPU_COUNTS}"
        return 0
    fi

    local detected
    detected="$(fabricperf_detect_gpus)"
    if [[ "${detected}" =~ ^[0-9]+$ && "${detected}" -gt 0 ]]; then
        local value
        for value in 2 4 8; do
            [[ "${value}" -le "${detected}" ]] && printf '%s ' "${value}"
        done
        echo
        return 0
    fi
    echo "2 4 8"
}

fabricperf_device_impl_values() {
    echo "${FABRICPERF_BENCH_DEVICE_IMPLS:-6 7 8}"
}

fabricperf_mode_values() {
    echo "${FABRICPERF_BENCH_MODES:-latency memory throughput}"
}

fabricperf_collective_binary_name() {
    case "$1" in
        sendrecv) echo "sendrecv_perf" ;;
        allgather|all_gather) echo "all_gather_perf" ;;
        reducescatter|reduce_scatter) echo "reduce_scatter_perf" ;;
        allreduce|all_reduce) echo "all_reduce_perf" ;;
        alltoall|all_to_all) echo "alltoall_perf" ;;
        *) echo "$1" ;;
    esac
}

fabricperf_find_nccl_test() {
    local collective="$1"
    local binary
    binary="$(fabricperf_collective_binary_name "${collective}")"

    if [[ -n "${FABRICPERF_BENCH_NCCL_TEST_BIN:-}" ]]; then
        [[ -x "${FABRICPERF_BENCH_NCCL_TEST_BIN}" ]] || fabricperf_die "not executable: ${FABRICPERF_BENCH_NCCL_TEST_BIN}"
        echo "${FABRICPERF_BENCH_NCCL_TEST_BIN}"
        return 0
    fi
    if command -v "${binary}" >/dev/null 2>&1; then
        command -v "${binary}"
        return 0
    fi

    local candidate
    for candidate in \
        "${NCCL_TESTS_BUILD:-}" \
        "${FABRICPERF_WORKSPACE_ROOT}/nccl-tests/build" \
        "${FABRICPERF_REPO_ROOT}/../nccl-tests/build"; do
        if [[ -n "${candidate}" && -x "${candidate}/${binary}" ]]; then
            echo "${candidate}/${binary}"
            return 0
        fi
    done

    candidate="$(find "${FABRICPERF_WORKSPACE_ROOT}" -path "*/nccl-tests/build/${binary}" -type f -executable -print -quit 2>/dev/null || true)"
    [[ -n "${candidate}" ]] || fabricperf_die "could not find ${binary}; set NCCL_TESTS_BUILD or FABRICPERF_BENCH_NCCL_TEST_BIN"
    echo "${candidate}"
}

fabricperf_find_mpirun() {
    if [[ -n "${MPIRUN:-}" ]]; then
        [[ -x "${MPIRUN}" ]] || fabricperf_die "not executable: ${MPIRUN}"
        echo "${MPIRUN}"
        return 0
    fi
    if command -v mpirun >/dev/null 2>&1; then
        command -v mpirun
        return 0
    fi
    local candidate="/usr/mpi/gcc/openmpi-4.1.9a1/bin/mpirun"
    [[ -x "${candidate}" ]] || fabricperf_die "could not find mpirun; set MPIRUN"
    echo "${candidate}"
}

fabricperf_build_sm_occupy() {
    # Build the optional SM occupancy sidecar on demand.
    # Example: FABRICPERF_BENCH_SM_OCCUPY=1 triggers this before nccl-tests.
    make -C "${FABRICPERF_PLUGIN_DIR}" sm-occupy >/dev/null
    local binary="${FABRICPERF_PLUGIN_DIR}/build/fabricperf_sm_occupy"
    [[ -x "${binary}" ]] || fabricperf_die "SM occupancy sidecar did not build: ${binary}"
    echo "${binary}"
}

fabricperf_start_sm_occupy() {
    # Start the optional SM occupancy sidecar and echo its PID.
    # Example: target 124-131 occupies the high-H100 SMID band during SendRecv.
    local trace_dir="$1"
    local binary
    binary="$(fabricperf_build_sm_occupy)"
    local target="${FABRICPERF_BENCH_SM_OCCUPY_TARGET:-124-131}"
    local duration_ms="${FABRICPERF_BENCH_SM_OCCUPY_MS:-30000}"
    local blocks="${FABRICPERF_BENCH_SM_OCCUPY_BLOCKS:-4096}"
    local threads="${FABRICPERF_BENCH_SM_OCCUPY_THREADS:-128}"
    local shared_kb="${FABRICPERF_BENCH_SM_OCCUPY_SHARED_KB:-200}"
    local devices="${FABRICPERF_BENCH_SM_OCCUPY_DEVICES:-all}"
    local warmup_ms="${FABRICPERF_BENCH_SM_OCCUPY_WARMUP_MS:-500}"
    local log_path="${trace_dir}/sm_occupy.log"

    echo "[fabricperf-bench] sm_occupy target=${target} duration_ms=${duration_ms} devices=${devices} log=${log_path}" >&2
    "${binary}" \
        --target-smids "${target}" \
        --duration-ms "${duration_ms}" \
        --blocks "${blocks}" \
        --threads "${threads}" \
        --shared-kb "${shared_kb}" \
        --devices "${devices}" \
        >"${log_path}" 2>&1 &
    local pid="$!"
    sleep "$(python3 - <<PY
print(max(0.0, float("${warmup_ms}") / 1000.0))
PY
)"
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
        wait "${pid}" || true
        fabricperf_die "SM occupancy sidecar exited before workload; see ${log_path}"
    fi
    echo "${pid}"
}

fabricperf_stop_sm_occupy() {
    # Stop the optional SM occupancy sidecar if it is still running.
    # Example: called after nccl-tests even when the workload returns nonzero.
    local pid="${1:-}"
    [[ -n "${pid}" ]] || return 0
    if kill -0 "${pid}" >/dev/null 2>&1; then
        kill "${pid}" >/dev/null 2>&1 || true
    fi
    wait "${pid}" >/dev/null 2>&1 || true
}

fabricperf_trace_dir() {
    local mode="$1"
    local collective="$2"
    local runtime="$3"
    local np="$4"
    local device_impl="${5:-}"
    local run_id="${FABRICPERF_BENCH_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
    # Store large trace trees on the workspace filesystem by default.
    # Example: FABRICPERF_BENCH_ROOT=/mnt/md127p1/home/cloud-user/songlin/payload/fabricperf_tmp/fabricperf_bench_1g
    # OLD: local root="${FABRICPERF_BENCH_ROOT:-/tmp/fabricperf_bench}"
    local root="${FABRICPERF_BENCH_ROOT:-${FABRICPERF_WORKSPACE_ROOT}/fabricperf_tmp/fabricperf_bench}"
    local name="${collective}_${runtime}_${mode}_np${np}"
    [[ -n "${device_impl}" ]] && name="${name}_d${device_impl}"
    echo "${FABRICPERF_TRACEDIR:-${root}/${name}_${run_id}}"
}

fabricperf_deepep_trace_dir() {
    local mode="$1"
    local run_id="${FABRICPERF_BENCH_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
    local root="${FABRICPERF_BENCH_ROOT:-${FABRICPERF_WORKSPACE_ROOT}/fabricperf_tmp/fabricperf_bench}"
    echo "${FABRICPERF_TRACEDIR:-${root}/deepep_ep8_${mode}_${run_id}}"
}

fabricperf_run_deepep_ep8() {
    local mode="${1:-${FABRICPERF_MODE:-latency}}"

    fabricperf_setup_paths

    local deepep_root
    local deepep_python
    deepep_root="$(fabricperf_find_deepep_root)"
    deepep_python="$(fabricperf_find_deepep_python)"

    local -a neutrino_cmd
    if [[ -n "${NEUTRINO_CMD:-}" ]]; then
        read -r -a neutrino_cmd <<< "${NEUTRINO_CMD}"
    elif command -v neutrino >/dev/null 2>&1; then
        neutrino_cmd=(neutrino)
    else
        neutrino_cmd=("${deepep_python}" -m neutrino.cli)
    fi

    local trace_dir
    trace_dir="$(fabricperf_deepep_trace_dir "${mode}")"
    mkdir -p "${trace_dir}"

    local exchange_dir="${FABRICPERF_EXCHANGE_DIR:-${trace_dir}/fabricperf_exchange}"
    local jit_dir="${EP_JIT_CACHE_DIR:-${trace_dir}/deepep_jit_cache}"
    local profile_dir="${DEEPEP_PROFILE_TRACE_DIR:-${trace_dir}/kineto}"
    mkdir -p "${exchange_dir}" "${jit_dir}" "${profile_dir}"

    local -a deepep_args=(
        "${FABRICPERF_BENCH_DIR}/deepep_ep8.py"
        --deepep-root "${deepep_root}"
        --num-processes "${DEEPEP_NUM_PROCESSES:-8}"
        # DeepEP EP8 docs list 24 SMs as the practical minimum; override with DEEPEP_NUM_SMS=32 when needed.
        --num-sms "${DEEPEP_NUM_SMS:-24}"
        --num-gpu-timeout-secs "${DEEPEP_GPU_TIMEOUT_SECS:-100}"
        --num-cpu-timeout-secs "${DEEPEP_CPU_TIMEOUT_SECS:-100}"
        --num-tokens "${DEEPEP_NUM_TOKENS:-4096}"
        --hidden "${DEEPEP_HIDDEN:-7168}"
        --num-topk "${DEEPEP_NUM_TOPK:-6}"
        --num-experts "${DEEPEP_NUM_EXPERTS:-256}"
        --test-first-only
        --ignore-local-traffic
        --dump-profile-traces "${profile_dir}"
    )
    if [[ -n "${DEEPEP_EXTRA_ARGS:-}" ]]; then
        local -a extra_args
        read -r -a extra_args <<< "${DEEPEP_EXTRA_ARGS}"
        deepep_args+=("${extra_args[@]}")
    fi

    local cuda_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    local -a env_args=(
        env
        "TMPDIR=${TMPDIR}"
        "PYTHONPATH=${deepep_root}${PYTHONPATH:+:${PYTHONPATH}}"
        "CUDA_VISIBLE_DEVICES=${cuda_devices}"
        "NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}"
        "EP_JIT_DUMP_PTX=${EP_JIT_DUMP_PTX:-1}"
        "EP_JIT_CACHE_DIR=${jit_dir}"
        "FABRICPERF_MODE=${mode}"
        "FABRICPERF_DEEPEP=1"
        "FABRICPERF_DEEPEP_LATENCY_MODE=${FABRICPERF_DEEPEP_LATENCY_MODE:-runtime}"
        "FABRICPERF_NCCL_DEVFUNC=0"
        "FABRICPERF_EXCHANGE_DIR=${exchange_dir}"
        "NEUTRINO_KERNEL=${NEUTRINO_KERNEL:-dispatch_impl:dispatch_copy_epilogue_impl:combine_impl:combine_reduce_epilogue_impl}"
    )
    if [[ "${mode}" == "throughput" ]]; then
        env_args+=("FABRICPERF_THROUGHPUT_PARTITIONS=${FABRICPERF_THROUGHPUT_PARTITIONS:-4}")
    fi
    echo "[fabricperf-bench] trace=${trace_dir}"
    echo "[fabricperf-bench] mode=${mode} deepep_root=${deepep_root} cuda=${cuda_devices}"
    # OLD: DeepEP was run directly, so FabricPerf children did not know rank/world.
    local -a command=("${env_args[@]}" "${neutrino_cmd[@]}" --plugin fabricperf --tracedir "${trace_dir}" \
        "${deepep_python}" "${deepep_args[@]}")
    if [[ -n "${FABRICPERF_BENCH_TIMEOUT:-}" ]]; then
        echo "[fabricperf-bench] timeout=${FABRICPERF_BENCH_TIMEOUT}"
        timeout "${FABRICPERF_BENCH_TIMEOUT}" "${command[@]}"
    else
        "${command[@]}"
    fi
}

fabricperf_run() {
    local mode="$1"
    local collective="$2"
    local runtime="$3"
    local np="${4:-$(fabricperf_default_np 2)}"
    local device_impl="${5:-8}"

    fabricperf_setup_paths

    local binary
    local mpirun
    binary="$(fabricperf_find_nccl_test "${collective}")"
    mpirun="$(fabricperf_find_mpirun)"

    local -a neutrino_cmd
    if [[ -n "${NEUTRINO_CMD:-}" ]]; then
        read -r -a neutrino_cmd <<< "${NEUTRINO_CMD}"
    elif command -v neutrino >/dev/null 2>&1; then
        neutrino_cmd=(neutrino)
    else
        neutrino_cmd=(python3 -m neutrino.cli)
    fi

    local -a runtime_args
    local devfunc="${FABRICPERF_BENCH_NCCL_DEVFUNC:-0}"
    case "${runtime}" in
        nccl)
            runtime_args=(-R 0)
            devfunc="${FABRICPERF_BENCH_NCCL_DEVFUNC:-1}"
            ;;
        dev|device)
            runtime_args=(-R 2 -D "${device_impl}")
            ;;
        *)
            fabricperf_die "runtime must be nccl or dev, got ${runtime}"
            ;;
    esac

    local -a test_args=(
        -b "${FABRICPERF_BENCH_MIN_BYTES:-1M}"
        -e "${FABRICPERF_BENCH_MAX_BYTES:-1M}"
        -f "${FABRICPERF_BENCH_FACTOR:-2}"
        -g "${FABRICPERF_BENCH_GPUS_PER_RANK:-1}"
        -n "${FABRICPERF_BENCH_ITERS:-10}"
        -w "${FABRICPERF_BENCH_WARMUPS:-2}"
    )
    if [[ -n "${FABRICPERF_BENCH_EXTRA_ARGS:-}" ]]; then
        local -a extra_args
        read -r -a extra_args <<< "${FABRICPERF_BENCH_EXTRA_ARGS}"
        test_args+=("${extra_args[@]}")
    fi

    local -a mpirun_args
    if [[ -n "${FABRICPERF_BENCH_MPIRUN_ARGS:-}" ]]; then
        read -r -a mpirun_args <<< "${FABRICPERF_BENCH_MPIRUN_ARGS}"
    else
        mpirun_args=()
    fi

    local trace_dir
    trace_dir="$(fabricperf_trace_dir "${mode}" "${collective}" "${runtime}" "${np}" "$([[ "${runtime}" == "nccl" ]] && echo "" || echo "${device_impl}")")"
    mkdir -p "$(dirname "${trace_dir}")"

    local -a env_args=(env "TMPDIR=${TMPDIR}" "FABRICPERF_MODE=${mode}" "FABRICPERF_NCCL_DEVFUNC=${devfunc}")
    if [[ "${mode}" == "throughput" ]]; then
        env_args+=("FABRICPERF_THROUGHPUT_PARTITIONS=${FABRICPERF_THROUGHPUT_PARTITIONS:-4}")
    fi
    local sm_occupy_pid=""
    if [[ "${FABRICPERF_BENCH_SM_OCCUPY:-0}" == "1" ]]; then
        mkdir -p "${trace_dir}"
        sm_occupy_pid="$(fabricperf_start_sm_occupy "${trace_dir}")"
    fi

    echo "[fabricperf-bench] trace=${trace_dir}"
    echo "[fabricperf-bench] mode=${mode} collective=${collective} runtime=${runtime} np=${np} device_impl=${device_impl}"
    # OLD: direct execution without a per-cell timeout made matrix scripts stop on a hung GPU workload.
    local -a command=("${env_args[@]}" "${neutrino_cmd[@]}" --plugin fabricperf --tracedir "${trace_dir}" \
        "${mpirun}" -np "${np}" "${mpirun_args[@]}" \
        "${binary}" "${test_args[@]}" "${runtime_args[@]}")
    local status=0
    if [[ -n "${FABRICPERF_BENCH_TIMEOUT:-}" ]]; then
        echo "[fabricperf-bench] timeout=${FABRICPERF_BENCH_TIMEOUT}"
        timeout "${FABRICPERF_BENCH_TIMEOUT}" "${command[@]}" || status=$?
    else
        "${command[@]}" || status=$?
    fi
    fabricperf_stop_sm_occupy "${sm_occupy_pid}"
    return "${status}"
}

fabricperf_run_collective_matrix_case() {
    # Run one matrix cell while optionally preserving later cells after a failure.
    local status=0
    fabricperf_run "$@" || status=$?
    if [[ "${status}" -eq 0 ]]; then
        return 0
    fi
    echo "[fabricperf-bench][warn] matrix cell failed status=${status}: $*" >&2
    if [[ "${FABRICPERF_BENCH_CONTINUE_ON_ERROR:-0}" == "1" ]]; then
        return 0
    fi
    return "${status}"
}

fabricperf_run_collective_matrix() {
    local collective="$1"
    local np
    local mode
    local device_impl
    for np in $(fabricperf_scale_values); do
        for mode in $(fabricperf_mode_values); do
            # OLD: fabricperf_run "${mode}" "${collective}" nccl "${np}"
            fabricperf_run_collective_matrix_case "${mode}" "${collective}" nccl "${np}"
            for device_impl in $(fabricperf_device_impl_values); do
                # OLD: fabricperf_run "${mode}" "${collective}" dev "${np}" "${device_impl}"
                fabricperf_run_collective_matrix_case "${mode}" "${collective}" dev "${np}" "${device_impl}"
            done
        done
    done
}
