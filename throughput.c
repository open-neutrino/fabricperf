#define FABRICPERF_NO_MPI 1
#include "common.h"

#include <stdbool.h>
#include <stdlib.h>

/* Throughput workgroup partition count; current probe layout accepts 4. */
#define FABRICPERF_THROUGHPUT_PARTITIONS_ENV "FABRICPERF_THROUGHPUT_PARTITIONS"

/* CUDA driver calls needed only for generated-module symbol writes. */
typedef struct throughput_cuda_driver {
    void* handle; /* dlopen handle for api->real_cuda_driver_path. */
    CUresult (*pfn_cuMemcpyHtoD)(CUdeviceptr dst, const void* src, size_t bytes); /* Legacy HtoD copy fallback. */
    CUresult (*pfn_cuMemcpyHtoD_v2)(CUdeviceptr dst, const void* src, size_t bytes); /* Preferred 64-bit HtoD copy. */
    CUresult (*pfn_cuModuleGetGlobal)(CUdeviceptr* dptr, size_t* bytes, CUmodule module, const char* name); /* Legacy global lookup. */
    CUresult (*pfn_cuModuleGetGlobal_v2)(CUdeviceptr* dptr, size_t* bytes, CUmodule module, const char* name); /* Preferred global lookup. */
} throughput_cuda_driver_t;

/* Neutrino ABI table retained for logging and real-driver discovery. */
static const neutrino_plugin_api_v1* api = NULL;
/* Process-local CUDA driver table for module symbol writes. */
static throughput_cuda_driver_t driver = {0};

/* Open libcuda and resolve the tiny symbol-write surface needed by throughput mode. */
static int load_cuda_driver(void) {
    if (driver.handle != NULL) {
        return 0;
    }
    if (api == NULL || api->real_cuda_driver_path == NULL || api->real_cuda_driver_path[0] == '\0') {
        PLUGIN_FAIL(-1, "did not receive a real CUDA driver path from Neutrino.");
    }

    driver.handle = dlopen(api->real_cuda_driver_path, RTLD_NOW | RTLD_LOCAL);
    if (driver.handle == NULL) {
        PLUGIN_FAIL(-1, "failed to load CUDA driver %s: %s",
                    api->real_cuda_driver_path, dlerror());
    }

    /* Optional v1/v2 pairs are checked below as at-least-one requirements. */
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuMemcpyHtoD, "cuMemcpyHtoD", false, -1);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuMemcpyHtoD_v2, "cuMemcpyHtoD_v2", false, -1);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuModuleGetGlobal, "cuModuleGetGlobal", false, -1);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuModuleGetGlobal_v2, "cuModuleGetGlobal_v2", false, -1);

    if (driver.pfn_cuMemcpyHtoD == NULL && driver.pfn_cuMemcpyHtoD_v2 == NULL) {
        PLUGIN_FAIL(-1, "could not resolve cuMemcpyHtoD.");
    }
    if (driver.pfn_cuModuleGetGlobal == NULL && driver.pfn_cuModuleGetGlobal_v2 == NULL) {
        PLUGIN_FAIL(-1, "could not resolve cuModuleGetGlobal.");
    }
    return 0;
}

/* Copy one value to a generated CUDA module global. */
static CUresult cuda_memcpy_htod(CUdeviceptr dst, const void* src, size_t bytes) {
    if (driver.pfn_cuMemcpyHtoD_v2 != NULL) {
        return driver.pfn_cuMemcpyHtoD_v2(dst, src, bytes);
    }
    return driver.pfn_cuMemcpyHtoD(dst, src, bytes);
}

/* Resolve one generated CUDA module global. */
static CUresult cuda_module_get_global(CUmodule module, const char* name, CUdeviceptr* dptr, size_t* bytes) {
    if (driver.pfn_cuModuleGetGlobal_v2 != NULL) {
        return driver.pfn_cuModuleGetGlobal_v2(dptr, bytes, module, name);
    }
    return driver.pfn_cuModuleGetGlobal(dptr, bytes, module, name);
}

/* Write a scalar symbol value and report hard failures to the caller. */
static int write_symbol(CUmodule module, const char* name, const void* data, size_t bytes) {
    if (module == 0 || name == NULL || data == NULL || bytes == 0) {
        return 0;
    }

    CUdeviceptr dst_ptr = 0;
    size_t dst_bytes = 0;
    if (cuda_module_get_global(module, name, &dst_ptr, &dst_bytes) != CUDA_SUCCESS) {
        PLUGIN_LOG("could not find throughput symbol %s", name);
        return -1;
    }
    if (bytes > dst_bytes) {
        PLUGIN_LOG("throughput symbol %s is too small (%zu < %zu)", name, dst_bytes, bytes);
        return -1;
    }
    if (cuda_memcpy_htod(dst_ptr, data, bytes) != CUDA_SUCCESS) {
        PLUGIN_LOG("could not write throughput symbol %s", name);
        return -1;
    }
    return 1;
}

/* Convert FABRICPERF_THROUGHPUT_PARTITIONS to the workgroup count. */
static uint32_t selected_partitions(void) {
    const char* raw = getenv(FABRICPERF_THROUGHPUT_PARTITIONS_ENV);
    if (raw == NULL || raw[0] == '\0') {
        return FABRICPERF_THROUGHPUT_PARTITIONS_DEFAULT;
    }
    if (strcmp(raw, "4") == 0) {
        return 4;
    }
    PLUGIN_LOG("ignoring invalid fixed-workgroup %s=%s; using %u",
               FABRICPERF_THROUGHPUT_PARTITIONS_ENV, raw,
               FABRICPERF_THROUGHPUT_PARTITIONS_DEFAULT);
    return FABRICPERF_THROUGHPUT_PARTITIONS_DEFAULT;
}

/* Read the first integer-valued launcher environment variable in a list. */
static int selected_env_int(const char* const* names, int fallback) {
    for (size_t idx = 0; names[idx] != NULL; idx++) {
        const char* raw = getenv(names[idx]);
        if (raw == NULL || raw[0] == '\0') {
            continue;
        }
        char* end = NULL;
        long value = strtol(raw, &end, 10);
        if (end != raw) {
            return (int) value;
        }
    }
    return fallback;
}

/* Required Neutrino plugin entry point for throughput runtime setup. */
int neutrino_plugin_init_v1(const neutrino_plugin_api_v1* plugin_api) {
    api = plugin_api;
    PLUGIN_REQUIRE_API(-1);
    if (load_cuda_driver() != 0) {
        return -1;
    }
    const char* const rank_envs[] = {
        "FABRICPERF_RANK",
        "OMPI_COMM_WORLD_RANK",
        "PMIX_RANK",
        "PMI_RANK",
        "SLURM_PROCID",
        NULL,
    };
    const char* const world_envs[] = {
        "OMPI_COMM_WORLD_SIZE",
        "PMIX_SIZE",
        "PMI_SIZE",
        "SLURM_NTASKS",
        NULL,
    };
    const char* const local_envs[] = {
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "PMIX_LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        NULL,
    };
    PLUGIN_LOG("prepared rank=%d world=%d device=%d",
               selected_env_int(rank_envs, 0),
               selected_env_int(world_envs, 1),
               selected_env_int(local_envs, 0));
    PLUGIN_LOG("initialized throughput runtime slots=%u bin_ns=%u partitions=%u",
               FABRICPERF_THROUGHPUT_SLOTS,
               FABRICPERF_THROUGHPUT_BIN_NS,
               selected_partitions());
    return 0;
}

/* Optional cleanup hook releases the locally opened CUDA driver handle. */
void neutrino_plugin_fini_v1(void) {
    if (driver.handle != NULL) {
        dlclose(driver.handle);
        memset(&driver, 0, sizeof(driver));
    }
}

/* Resolve throughput-owned module globals and leave all other symbols to Neutrino. */
int neutrino_plugin_resolve_symbol_v1(const neutrino_plugin_symbol_v1* symbol,
                                      void* destination_module,
                                      const neutrino_plugin_module_context_v1* context) {
    (void) context;
    if (symbol == NULL || symbol->name == NULL) {
        return 0;
    }

    CUmodule destination = (CUmodule) destination_module;
    if (PLUGIN_IS_SYMBOL(symbol->name, "fabricperfThroughputSlots")) {
        uint32_t value = (uint32_t) FABRICPERF_THROUGHPUT_SLOTS;
        return write_symbol(destination, symbol->name, &value, sizeof(value));
    }
    if (PLUGIN_IS_SYMBOL(symbol->name, "fabricperfThroughputBinNs")) {
        uint32_t value = (uint32_t) FABRICPERF_THROUGHPUT_BIN_NS;
        return write_symbol(destination, symbol->name, &value, sizeof(value));
    }
    if (PLUGIN_IS_SYMBOL(symbol->name, "fabricperfThroughputPartitions")) {
        uint32_t value = selected_partitions();
        return write_symbol(destination, symbol->name, &value, sizeof(value));
    }
    if (PLUGIN_IS_SYMBOL(symbol->name, "launchIndex")) {
        uint32_t value = context != NULL ? context->launch_index : 0;
        return write_symbol(destination, symbol->name, &value, sizeof(value));
    }
    return 0;
}
