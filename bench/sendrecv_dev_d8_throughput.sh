#!/usr/bin/env bash
# Profile nccl-tests sendrecv device kernels (-R 2 -D 8) with FabricPerf throughput mode.

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

fabricperf_run throughput sendrecv dev "" 8
