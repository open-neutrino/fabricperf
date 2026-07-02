NEUTRINO_SRC ?= ../../src
CUDA_HOME ?= /usr/local/cuda
CUDA_INC ?= $(CUDA_HOME)/targets/x86_64-linux/include
CUDA_LIB ?= $(CUDA_HOME)/targets/x86_64-linux/lib
CUDA_DRIVER_LIB ?= /usr/lib/x86_64-linux-gnu
MPI_HOME ?= /usr/mpi/gcc/openmpi-4.1.9a1
NVCC ?= $(CUDA_HOME)/bin/nvcc

MPI_CC := $(shell command -v mpicc 2>/dev/null)
ifeq ($(MPI_CC),)
MPI_CC := $(wildcard $(MPI_HOME)/bin/mpicc)
endif

MPI_CXX := $(shell command -v mpicxx 2>/dev/null)
ifeq ($(MPI_CXX),)
MPI_CXX := $(shell command -v mpic++ 2>/dev/null)
endif
ifeq ($(MPI_CXX),)
MPI_CXX := $(wildcard $(MPI_HOME)/bin/mpicxx)
endif

USER_CC_ORIGIN := $(origin CC)
# GNU make defaults CC to cc; FabricPerf needs MPI symbols, so use mpicc by default.
ifeq ($(origin CC), default)
CC := $(if $(MPI_CC),$(MPI_CC),mpicc)
endif
ifeq ($(USER_CC_ORIGIN),default)
THROUGHPUT_CC ?= cc
else
THROUGHPUT_CC ?= $(CC)
endif
CFLAGS ?= -O3 -fPIC -shared -Wall -Wextra
# GNU make defaults CXX to g++; CUPTI one-pass mode also needs MPI symbols, so use mpicxx when available.
ifeq ($(origin CXX), default)
CXX := $(if $(MPI_CXX),$(MPI_CXX),g++)
endif
CXXFLAGS ?= -O3 -fPIC -shared -Wall -Wextra -Wno-missing-field-initializers -std=c++17

FABRICPERF_MODE ?= latency
VALID_FABRICPERF_MODES := latency memory throughput
ifneq ($(filter $(FABRICPERF_MODE),$(VALID_FABRICPERF_MODES)),$(FABRICPERF_MODE))
$(error FABRICPERF_MODE must be one of: $(VALID_FABRICPERF_MODES))
endif

ifeq ($(FABRICPERF_CUPTI),1)
# Memory mode still uses CUPTI/NVPerf; example: `FABRICPERF_CUPTI=1 make`.
PLUGIN_SRC := memory.cc
PLUGIN_LINK = $(CXX) $(CXXFLAGS) -I$(NEUTRINO_SRC) -I$(CUDA_INC) $< -o $@ \
	-L$(CUDA_LIB) -Wl,-rpath,$(CUDA_LIB) -lcupti -lnvperf_host -lnvperf_target -ldl -lpthread
else ifeq ($(FABRICPERF_MODE),memory)
# Direct memory-mode builds mirror the CLI's FABRICPERF_CUPTI selector.
PLUGIN_SRC := memory.cc
PLUGIN_LINK = $(CXX) $(CXXFLAGS) -I$(NEUTRINO_SRC) -I$(CUDA_INC) $< -o $@ \
	-L$(CUDA_LIB) -Wl,-rpath,$(CUDA_LIB) -lcupti -lnvperf_host -lnvperf_target -ldl -lpthread
else ifeq ($(FABRICPERF_MODE),throughput)
# Throughput mode uses a standalone CUDA-symbol runtime with no MPI/CUPTI setup.
PLUGIN_SRC := throughput.c
PLUGIN_LINK = $(THROUGHPUT_CC) $(CFLAGS) -I$(NEUTRINO_SRC) -I$(CUDA_INC) $< -o $@ -ldl -lpthread
else
# Latency remains the default FabricPerf runtime.
PLUGIN_SRC := latency.c
PLUGIN_LINK = $(CC) $(CFLAGS) -I$(NEUTRINO_SRC) -I$(CUDA_INC) $< -o $@ -ldl -lpthread
endif

all: build/fabricperf_plugin.so

# Build the runtime plugin against Neutrino's stable plugin ABI header.
build/fabricperf_plugin.so: $(PLUGIN_SRC) common.h $(NEUTRINO_SRC)/plugin.h FORCE
	mkdir -p build
	$(PLUGIN_LINK)

# Build a standalone MPI/CUDA fabric-handle diagnostic; example:
# `make -C neutrino/plugins/fabricperf fabric-handle-test`.
fabric-handle-test: build/fabric_handle_mpi_test

# Link against libcuda directly because this test is independent of Neutrino.
build/fabric_handle_mpi_test: fabric_handle_mpi_test.cc FORCE
	mkdir -p build
	$(CXX) -O2 -Wall -Wextra -std=c++17 -I$(CUDA_INC) $< -o $@ \
		-L$(CUDA_DRIVER_LIB) -Wl,-rpath,$(CUDA_DRIVER_LIB) -lcuda

# Build the optional SM occupancy sidecar; example:
# `make -C neutrino/plugins/fabricperf sm-occupy`.
sm-occupy: build/fabricperf_sm_occupy

build/fabricperf_sm_occupy: bench/sm_occupy.cu FORCE
	mkdir -p build
	$(NVCC) -O2 -std=c++17 $< -o $@

clean:
	rm -rf build

FORCE:

.PHONY: all fabric-handle-test sm-occupy clean FORCE
