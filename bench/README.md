# FabricPerf Bench Scripts

These scripts run nccl-tests through Neutrino's FabricPerf plugin. They detect
the checkout, Python path, `mpirun`, NCCL library directory, and nccl-tests
binary when possible.

Run them outside the sandbox because they launch CUDA and MPI. NCCL `-R 0`
scripts enable `FABRICPERF_NCCL_DEVFUNC=1` by default; device-kernel scripts use
`FABRICPERF_NCCL_DEVFUNC=0`.

The six sendrecv smoke scripts cover three FabricPerf modes over two sendrecv
implementations:

```sh
./neutrino/plugins/fabricperf/bench/sendrecv_nccl_latency.sh
./neutrino/plugins/fabricperf/bench/sendrecv_nccl_memory.sh
./neutrino/plugins/fabricperf/bench/sendrecv_nccl_throughput.sh
./neutrino/plugins/fabricperf/bench/sendrecv_dev_d8_latency.sh
./neutrino/plugins/fabricperf/bench/sendrecv_dev_d8_memory.sh
./neutrino/plugins/fabricperf/bench/sendrecv_dev_d8_throughput.sh
```

Collective matrix scripts cover `sendrecv`, `allgather`, `reducescatter`,
`allreduce`, and `alltoall`. By default they run scales detected from
`nvidia-smi` up to `2 4 8`, all three FabricPerf modes, NCCL `-R 0`, and
device-kernel `-R 2 -D 6/7/8`:

```sh
./neutrino/plugins/fabricperf/bench/sendrecv_matrix.sh
./neutrino/plugins/fabricperf/bench/allgather_matrix.sh
./neutrino/plugins/fabricperf/bench/reducescatter_matrix.sh
./neutrino/plugins/fabricperf/bench/allreduce_matrix.sh
./neutrino/plugins/fabricperf/bench/alltoall_matrix.sh
```

DeepEP V2 EP8 scale-up profiling uses the official DeepEP elastic benchmark
through a FabricPerf rank-env wrapper. It defaults to GPUs `0-7`,
`--num-processes 8`, `--test-first-only`, `--ignore-local-traffic`,
`NCCL_NVLS_ENABLE=0`, and `EP_JIT_DUMP_PTX=1`. Latency mode uses
`FABRICPERF_EXCHANGE_DIR` for non-MPI CUDA fabric-handle exchange before the
kernel mailbox probes run:

```sh
./neutrino/plugins/fabricperf/bench/deepep_ep8.sh latency
./neutrino/plugins/fabricperf/bench/deepep_ep8.sh memory
./neutrino/plugins/fabricperf/bench/deepep_ep8.sh throughput
./neutrino/plugins/fabricperf/bench/deepep_ep8_matrix.sh
```

Useful overrides:

```sh
NCCL_TESTS_BUILD=/path/to/nccl-tests/build
NCCL_LIB_DIR=/path/to/nccl/build/lib
MPIRUN=/path/to/mpirun
NEUTRINO_CMD="python3 -m neutrino.cli"
# Default: <payload-checkout>/fabricperf_tmp/tmp
# OLD: TMPDIR was usually unset, so Python/tools used /tmp.
TMPDIR=/mnt/md127p1/home/cloud-user/songlin/payload/fabricperf_tmp/tmp
# Default: <payload-checkout>/fabricperf_tmp/fabricperf_bench
# OLD: FABRICPERF_BENCH_ROOT=/tmp/fabricperf_bench
FABRICPERF_BENCH_ROOT=/mnt/md127p1/home/cloud-user/songlin/payload/fabricperf_tmp/fabricperf_bench_1g
FABRICPERF_BENCH_GPU_COUNTS="2 4 8"
FABRICPERF_BENCH_DEVICE_IMPLS="6 7 8"
FABRICPERF_BENCH_MODES="latency memory throughput"
FABRICPERF_BENCH_MIN_BYTES=1M
FABRICPERF_BENCH_MAX_BYTES=1M
FABRICPERF_BENCH_ITERS=10
FABRICPERF_BENCH_WARMUPS=2
FABRICPERF_BENCH_EXTRA_ARGS="--check 0"
FABRICPERF_BENCH_MPIRUN_ARGS="--bind-to none"
FABRICPERF_BENCH_TIMEOUT=20m
FABRICPERF_BENCH_CONTINUE_ON_ERROR=1
DEEPEP_ROOT=/path/to/DeepEP
DEEPEP_PYTHON=/path/to/.venv-deepep/bin/python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
DEEPEP_EXTRA_ARGS="--skip-check"
```
