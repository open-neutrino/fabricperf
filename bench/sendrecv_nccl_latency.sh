#!/usr/bin/env bash
# Profile nccl-tests sendrecv through NCCL kernels (-R 0) with FabricPerf latency mode.

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

fabricperf_run latency sendrecv nccl
