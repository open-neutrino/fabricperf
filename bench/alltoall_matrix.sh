#!/usr/bin/env bash
# Run FabricPerf modes over alltoall_perf for NCCL and device-kernel variants.

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

fabricperf_run_collective_matrix alltoall
