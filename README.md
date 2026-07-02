# FabricPerf 

by Songlin Huang and Chenshu Wu from the University of Hong Kong. 

FabricPerf is a Scale-up Network Measurement tools based on [neutrino](https://github.com/open-neutrino/neutrino), 
supporting three modes (selected by `FABRICPERF_MODE`): 

- `latency` analyze PTP-corrected send/recv latencies.
- `throughput` analyze intra-kernel throughput variations. 
- `memory` analyze DRAM/LLC behavior with help from CUPTI. 

## Prerequisites

- CUDA driver support, typically headers under `/usr/local/cuda/targets/x86_64-linux/include`.
- MPI compiler and runtime, defaulting to `mpicc`.
- Neutrino source code from Github

Build with: 
```sh
make -C neutrino/plugins/fabricperf NEUTRINO_SRC=../../src CUDA_INC=/path/to/cuda/include
```

## Usage

Latency mode is the default:

```sh
FABRICPERF_MODE=latency/throughput/memory neutrino --plugin fabricperf --tracedir ./trace 
  mpirun -np 2 ./sendrecv_workload
```

Note: 
1. All communications must be run with `mpirun`. 
2. For unrelated kernels, use `--kernel` / `--filter` (or `NEUTRINO_KERNEL` /
`NEUTRINO_FILTER`) to filter them out.

## Output

Latency mode prints one row per trace rank and channel:

```text
trace  rank  src  channel  id  count  invalid  invalid_pct  avg_us  min_us  p50_us  max_us  steps
```

`avg_us`, `min_us`, `p50_us`, and `max_us` are microseconds derived
from GPU globaltimer ticks in `sr_latency` records. `channel` is the record
field used for grouping, currently `vblock` or `cta` depending on the generated
trace schema. `steps` is the inclusive observed send/recv step range.

Memory mode writes `fabricperf_cupti.csv`:

```text
rank,device,launch_index,kernel,grid,block,shared_bytes,duration_s,dram_read_Bps,dram_write_Bps,nvlink_rx_Bps,nvlink_tx_Bps,xbar_read_Bps,xbar_write_Bps,xbar_metric,xbar_value,raw_metrics
```

`dram_read_Bps`, `dram_write_Bps`, `xbar_read_Bps`, and `xbar_write_Bps` are
derived from Neutrino probe byte counts divided by the PM/event duration.
`nvlink_rx_Bps` and `nvlink_tx_Bps` come from CUPTI PM Sampling. `raw_metrics`
records PM Sampling metadata plus `probe_ld_global_bytes`,
`probe_st_global_bytes`, `probe_cp_async_bytes`, `probe_read_bytes`, and
`probe_write_bytes`.

Throughput mode prints one row per trace rank and workgroup:

```text
trace rank variant workgroup captured dropped duration_ticks avg_pps peak_bin_pps bin_ns
```

## Configurations

FabricPerf public runtime controls:

```text
FABRICPERF_MODE=latency|memory|throughput
FABRICPERF_THROUGHPUT_PARTITIONS=4
FABRICPERF_NCCL_DEVFUNC=1
FABRICPERF_RANK
FABRICPERF_HOSTID
```

`FABRICPERF_THROUGHPUT_PARTITIONS` is part of the throughput trace contract and
currently must be unset or `4`; future probe layouts can widen this to `1`,
`2`, `4`, or more without changing the top-level mode. `FABRICPERF_RANK` and
`FABRICPERF_HOSTID` override rank/host naming when the MPI runtime does not
provide a usable rank.
`FABRICPERF_NCCL_DEVFUNC=1` selects the FabricPerf-local NCCL devFunc prober
wrapper; target function and funcId selection are automatic.
Throughput CSV rows still report `variant=gmem` for compatibility, but there is
no longer a public or internal `FABRICPERF_THROUGHPUT_VARIANT` selector in the
active GMEM-only path.

Build-time controls:

```text
FABRICPERF_CUPTI=1
NEUTRINO_SRC
CUDA_HOME
CUDA_INC
CUDA_LIB
MPI_HOME
MPI_CC
CC
CXX
CFLAGS
CXXFLAGS
```

`FABRICPERF_CUPTI=1` is only a Makefile selector for the memory runtime. Runtime
behavior is selected with `FABRICPERF_MODE`.

## Citation

Please cite our work if you use FabricPerf:

```bibtex
@inproceedings{huang2026fabricperf,
  author = {Huang, Songlin and Wu, Chenshu},
  title = {FabricPerf: Measuring NIC-less Scale-Up Network through GPU Communication Kernel Profiling},
  booktitle = {Proceedings of the ACM SIGCOMM 2026 Conference},
  series = {SIGCOMM '26},
  year = {2026},
  location = {Denver, CO, USA},
  publisher = {ACM},
  address = {New York, NY, USA},
  numpages = {15},
  doi = {10.1145/3789240.3829172},
  url = {https://doi.org/10.1145/3789240.3829172},
}
```
