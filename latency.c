#include "common.h"

#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* File-based exchange env for non-MPI runtimes such as DeepEP torch.spawn. */
#define FABRICPERF_EXCHANGE_DIR_ENV "FABRICPERF_EXCHANGE_DIR"
/* Default setup wait for all rank fabric handles in file-exchange mode. */
#define FABRICPERF_EXCHANGE_TIMEOUT_MS_DEFAULT 120000
/* Optional timeout override for file-exchange mode. */
#define FABRICPERF_EXCHANGE_TIMEOUT_MS_ENV "FABRICPERF_EXCHANGE_TIMEOUT_MS"

/*
 * Mutable plugin runtime state.
 * This lives in the plugin, not Neutrino core, so MPI rank data and fabric VMM
 * handles never become part of the Neutrino ABI.
 */
typedef struct {
    bool mpi_prepared;          /* MPI rank/size have been queried successfully. */
    bool file_exchange;         /* Rank/handle exchange uses files instead of MPI. */
    bool buffers_prepared;      /* Fabric buffers and device tables are ready. */
    bool reported;              /* Preparation summary has been logged once. */
    bool ptp_scale_reported;    /* Unsupported PTP scale diagnostic was logged once. */
    int world_rank;             /* MPI_COMM_WORLD rank for this process. */
    int world_size;             /* MPI_COMM_WORLD size for table dimensions. */
    char exchange_dir[PATH_MAX]; /* Directory for non-MPI fabric-handle exchange. */
    CUdevice device;            /* Current CUDA device associated with context. */
    size_t leader_alloc_size;   /* Mapped byte size for each leader buffer. */
    size_t follower_alloc_size; /* Mapped byte size for each follower buffer. */
    size_t mailbox_alloc_size;  /* Mapped byte size for each latency mailbox. */
    CUmemGenericAllocationHandle* leader_handles;   /* VMM handles per rank. */
    CUmemGenericAllocationHandle* follower_handles; /* VMM handles per rank. */
    CUmemGenericAllocationHandle* mailbox_handles;  /* VMM handles per mailbox owner. */
    fabricperf_slot_t** leader_buffs;   /* Host table of mapped leader pointers. */
    fabricperf_slot_t** follower_buffs; /* Host table of mapped follower pointers. */
    fabricperf_slot_t** mailbox_buffs;  /* Host table of mapped per-rank mailboxes. */
    fabricperf_slot_t* local_leader_buff;   /* Local rank's leader mapping. */
    fabricperf_slot_t* local_follower_buff; /* Local rank's follower mapping. */
    fabricperf_slot_t* local_mailbox_buff;  /* Receiver-owned mailbox cleared per launch. */
    CUdeviceptr device_leader_table;    /* Device copy of leader_buffs. */
    CUdeviceptr device_follower_table;  /* Device copy of follower_buffs. */
    CUdeviceptr device_mailbox_table;   /* Device copy of mailbox_buffs. */
    CUdeviceptr device_result_buffer;   /* Device result scratch exposed to PTX. */
    CUdeviceptr device_latency_offset_buffer; /* Per-rank source-clock offset table. */
    size_t result_buffer_size;          /* Byte size of device_result_buffer. */
    size_t latency_offset_buffer_size;  /* Byte size of device_latency_offset_buffer. */
} fabricperf_runtime_t;

/* Small Neutrino ABI table retained after init; CUDA/MPI are resolved locally. */
static const neutrino_plugin_api_v1* api = NULL;
/* Protects lazy MPI and fabric-buffer preparation from concurrent launches. */
static pthread_mutex_t runtime_mutex = PTHREAD_MUTEX_INITIALIZER;
/* Single process-local runtime state instance for this plugin. */
static fabricperf_runtime_t runtime_state = {0};

/*
 * CUDA driver function table owned by FabricPerf.
 * Field names use pfn_ prefixes to avoid cuda.h macro aliases such as
 * cuMemcpyHtoD -> cuMemcpyHtoD_v2 rewriting struct member names.
 */
typedef struct {
    void* handle; /* dlopen handle for api->real_cuda_driver_path. */
    CUresult (*pfn_cuCtxSynchronize)(void); /* Drain work before cleanup. */
    CUresult (*pfn_cuCtxGetDevice)(CUdevice* device); /* Find active device. */
    CUresult (*pfn_cuDeviceGetAttribute)(int* value, CUdevice_attribute attrib, CUdevice device); /* Feature checks. */
    CUresult (*pfn_cuMemGetAllocationGranularity)(size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option); /* VMM alignment. */
    CUresult (*pfn_cuMemCreate)(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags); /* Local VMM allocation. */
    CUresult (*pfn_cuMemAddressReserve)(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags); /* VA reservation. */
    CUresult (*pfn_cuMemMap)(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags); /* Map VMM handle. */
    CUresult (*pfn_cuMemSetAccess)(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count); /* Device access rights. */
    CUresult (*pfn_cuMemExportToShareableHandle)(void* shareable_handle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handle_type, unsigned long long flags); /* Fabric export. */
    CUresult (*pfn_cuMemImportFromShareableHandle)(CUmemGenericAllocationHandle* handle, void* shareable_handle, CUmemAllocationHandleType handle_type); /* Fabric import. */
    CUresult (*pfn_cuMemUnmap)(CUdeviceptr ptr, size_t size); /* Unmap VA during cleanup. */
    CUresult (*pfn_cuMemAddressFree)(CUdeviceptr ptr, size_t size); /* Release reserved VA. */
    CUresult (*pfn_cuMemRelease)(CUmemGenericAllocationHandle handle); /* Release VMM handle. */
    CUresult (*pfn_cuMemAlloc_v2)(CUdeviceptr* ptr, size_t bytes); /* Allocate device tables/results. */
    CUresult (*pfn_cuMemFree_v2)(CUdeviceptr ptr); /* Free device tables/results. */
    CUresult (*pfn_cuMemsetD8)(CUdeviceptr ptr, unsigned char value, size_t bytes); /* Legacy memset fallback. */
    CUresult (*pfn_cuMemsetD8_v2)(CUdeviceptr ptr, unsigned char value, size_t bytes); /* Preferred memset. */
    CUresult (*pfn_cuMemcpyHtoD)(CUdeviceptr dst, const void* src, size_t bytes); /* Legacy HtoD fallback. */
    CUresult (*pfn_cuMemcpyHtoD_v2)(CUdeviceptr dst, const void* src, size_t bytes); /* Preferred HtoD copy. */
    CUresult (*pfn_cuModuleGetGlobal)(CUdeviceptr* dptr, size_t* bytes, CUmodule module, const char* name); /* Legacy symbol lookup. */
    CUresult (*pfn_cuModuleGetGlobal_v2)(CUdeviceptr* dptr, size_t* bytes, CUmodule module, const char* name); /* Preferred symbol lookup. */
} fabricperf_cuda_driver_t;

/* Resolved CUDA driver function table; zero means unresolved. */
static fabricperf_cuda_driver_t cuda_driver = {0};

/*
 * Open the real CUDA driver and resolve only the calls FabricPerf needs.
 * This replaces the old Neutrino ABI CUDA helper table and keeps driver
 * version choices inside the plugin.
 */
static int load_cuda_driver(void) {
    if (cuda_driver.handle != NULL) {
        return 0;
    }
    if (api == NULL || api->real_cuda_driver_path == NULL || api->real_cuda_driver_path[0] == '\0') {
        PLUGIN_FAIL(-1, "did not receive a real CUDA driver path from Neutrino.");
    }

    cuda_driver.handle = dlopen(api->real_cuda_driver_path, RTLD_NOW | RTLD_LOCAL);
    if (cuda_driver.handle == NULL) {
        PLUGIN_LOG("failed to load CUDA driver %s: %s",
                       api->real_cuda_driver_path, dlerror());
        return -1;
    }

    /* Required symbols are necessary for fabric VMM allocation and cleanup. */
    PLUGIN_LOAD_CUDA_SYMBOL(&cuda_driver, pfn_cuCtxSynchronize, "cuCtxSynchronize", true, -1);
    PLUGIN_LOAD_CUDA_SYMBOL(&cuda_driver, pfn_cuCtxGetDevice, "cuCtxGetDevice", true, -1);
    PLUGIN_LOAD_CUDA_SYMBOL(&cuda_driver, pfn_cuDeviceGetAttribute, "cuDeviceGetAttribute", true, -1);
    PLUGIN_LOAD_CUDA_SYMBOL(&cuda_driver, pfn_cuMemGetAllocationGranularity, "cuMemGetAllocationGranularity", true, -1);
    PLUGIN_LOAD_CUDA_SYMBOL(&cuda_driver, pfn_cuMemCreate, "cuMemCreate", true, -1);
    PLUGIN_LOAD_CUDA_SYMBOL(&cuda_driver, pfn_cuMemAddressReserve, "cuMemAddressReserve", true, -1);
    PLUGIN_LOAD_CUDA_SYMBOL(&cuda_driver, pfn_cuMemMap, "cuMemMap", true, -1);
    PLUGIN_LOAD_CUDA_SYMBOL(&cuda_driver, pfn_cuMemSetAccess, "cuMemSetAccess", true, -1);
    PLUGIN_LOAD_CUDA_SYMBOL(&cuda_driver, pfn_cuMemExportToShareableHandle, "cuMemExportToShareableHandle", true, -1);
    PLUGIN_LOAD_CUDA_SYMBOL(&cuda_driver, pfn_cuMemImportFromShareableHandle, "cuMemImportFromShareableHandle", true, -1);
    PLUGIN_LOAD_CUDA_SYMBOL(&cuda_driver, pfn_cuMemUnmap, "cuMemUnmap", true, -1);
    PLUGIN_LOAD_CUDA_SYMBOL(&cuda_driver, pfn_cuMemAddressFree, "cuMemAddressFree", true, -1);
    PLUGIN_LOAD_CUDA_SYMBOL(&cuda_driver, pfn_cuMemRelease, "cuMemRelease", true, -1);
    PLUGIN_LOAD_CUDA_SYMBOL(&cuda_driver, pfn_cuMemAlloc_v2, "cuMemAlloc_v2", true, -1);
    PLUGIN_LOAD_CUDA_SYMBOL(&cuda_driver, pfn_cuMemFree_v2, "cuMemFree_v2", true, -1);
    /* Optional v1/v2 pairs are validated as "at least one" after lookup. */
    PLUGIN_LOAD_CUDA_SYMBOL(&cuda_driver, pfn_cuMemsetD8, "cuMemsetD8", false, -1);
    PLUGIN_LOAD_CUDA_SYMBOL(&cuda_driver, pfn_cuMemsetD8_v2, "cuMemsetD8_v2", false, -1);
    PLUGIN_LOAD_CUDA_SYMBOL(&cuda_driver, pfn_cuMemcpyHtoD, "cuMemcpyHtoD", false, -1);
    PLUGIN_LOAD_CUDA_SYMBOL(&cuda_driver, pfn_cuMemcpyHtoD_v2, "cuMemcpyHtoD_v2", false, -1);
    PLUGIN_LOAD_CUDA_SYMBOL(&cuda_driver, pfn_cuModuleGetGlobal, "cuModuleGetGlobal", false, -1);
    PLUGIN_LOAD_CUDA_SYMBOL(&cuda_driver, pfn_cuModuleGetGlobal_v2, "cuModuleGetGlobal_v2", false, -1);

    if (cuda_driver.pfn_cuMemsetD8 == NULL && cuda_driver.pfn_cuMemsetD8_v2 == NULL) {
        PLUGIN_FAIL(-1, "could not resolve cuMemsetD8.");
    }
    if (cuda_driver.pfn_cuMemcpyHtoD == NULL && cuda_driver.pfn_cuMemcpyHtoD_v2 == NULL) {
        PLUGIN_FAIL(-1, "could not resolve cuMemcpyHtoD.");
    }
    if (cuda_driver.pfn_cuModuleGetGlobal == NULL && cuda_driver.pfn_cuModuleGetGlobal_v2 == NULL) {
        PLUGIN_FAIL(-1, "could not resolve cuModuleGetGlobal.");
    }
    return 0;
}

/* Prefer v2 memset when present, but tolerate older driver exports. */
static CUresult cuda_memset_d8(CUdeviceptr ptr, unsigned char value, size_t bytes) {
    if (cuda_driver.pfn_cuMemsetD8_v2 != NULL) {
        return cuda_driver.pfn_cuMemsetD8_v2(ptr, value, bytes);
    }
    return cuda_driver.pfn_cuMemsetD8(ptr, value, bytes);
}

/* Prefer v2 host-to-device copies for 64-bit CUDA driver ABI compatibility. */
static CUresult cuda_memcpy_htod(CUdeviceptr dst, const void* src, size_t bytes) {
    if (cuda_driver.pfn_cuMemcpyHtoD_v2 != NULL) {
        return cuda_driver.pfn_cuMemcpyHtoD_v2(dst, src, bytes);
    }
    return cuda_driver.pfn_cuMemcpyHtoD(dst, src, bytes);
}

/* Resolve a module global through the real CUDA driver, not through Neutrino. */
static CUresult cuda_module_get_global(CUmodule module, const char* name, CUdeviceptr* dptr, size_t* bytes) {
    if (cuda_driver.pfn_cuModuleGetGlobal_v2 != NULL) {
        return cuda_driver.pfn_cuModuleGetGlobal_v2(dptr, bytes, module, name);
    }
    return cuda_driver.pfn_cuModuleGetGlobal(dptr, bytes, module, name);
}

/* Generic symbol writer used by FabricPerf-specific resolve_symbol handling. */
static int symbol_write(CUmodule module, const char* name, const void* data, size_t bytes) {
    if (module == 0 || name == NULL || data == NULL || bytes == 0) {
        return 0;
    }
    CUdeviceptr dst_ptr = 0;
    size_t dst_bytes = 0;
    if (cuda_module_get_global(module, name, &dst_ptr, &dst_bytes) != CUDA_SUCCESS) {
        return 0;
    }
    size_t copy_bytes = dst_bytes < bytes ? dst_bytes : bytes;
    return cuda_memcpy_htod(dst_ptr, data, copy_bytes) == CUDA_SUCCESS ? 1 : 0;
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

/*
 * Prepare rank metadata for non-MPI runtimes.
 * DeepEP launches one Python multiprocessing worker per GPU; those workers do
 * not call MPI_Init, so FabricPerf uses env ranks plus file exchange for CUDA
 * fabric handles.
 */
static int prepare_file_exchange_locked(void) {
    const char* exchange_dir = getenv(FABRICPERF_EXCHANGE_DIR_ENV);
    if (exchange_dir == NULL || exchange_dir[0] == '\0') {
        // OLD: latency mode required MPI_Init before probed launches.
        PLUGIN_FAIL(-1, "requires MPI_Init or %s for non-MPI rank exchange.",
                    FABRICPERF_EXCHANGE_DIR_ENV);
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
        "FABRICPERF_WORLD_SIZE",
        "OMPI_COMM_WORLD_SIZE",
        "PMIX_SIZE",
        "PMI_SIZE",
        "SLURM_NTASKS",
        NULL,
    };
    runtime_state.world_rank = selected_env_int(rank_envs, -1);
    runtime_state.world_size = selected_env_int(world_envs, -1);
    if (runtime_state.world_rank < 0 || runtime_state.world_size < 1 ||
        runtime_state.world_rank >= runtime_state.world_size ||
        runtime_state.world_size > FABRICPERF_MAX_CHANNEL) {
        PLUGIN_FAIL(-1, "invalid non-MPI FabricPerf rank/world rank=%d world=%d.",
                    runtime_state.world_rank, runtime_state.world_size);
    }
    if (snprintf(runtime_state.exchange_dir, sizeof(runtime_state.exchange_dir),
                 "%s", exchange_dir) >= (int) sizeof(runtime_state.exchange_dir)) {
        PLUGIN_FAIL(-1, "%s is too long.", FABRICPERF_EXCHANGE_DIR_ENV);
    }
    if (mkdir(runtime_state.exchange_dir, 0775) != 0 && errno != EEXIST) {
        PLUGIN_FAIL(-1, "could not create %s=%s.",
                    FABRICPERF_EXCHANGE_DIR_ENV,
                    runtime_state.exchange_dir);
    }

    runtime_state.file_exchange = true;
    runtime_state.mpi_prepared = true;
    PLUGIN_LOG("prepared non-MPI rank=%d world=%d exchange=%s",
               runtime_state.world_rank,
               runtime_state.world_size,
               runtime_state.exchange_dir);
    return 0;
}

/*
 * Lazily validate MPI and cache COMM_WORLD rank/size.
 * FabricPerf needs MPI only for probes that request rank-aware symbols, so
 * non-FabricPerf probes can still load the plugin without MPI_Init.
 */
static int prepare_mpi_locked(void) {
    if (runtime_state.mpi_prepared) {
        return 0;
    }

    int mpi_initialized = 0;
    /* MPI is owned directly by the plugin; Neutrino core has no MPI shim. */
    PLUGIN_CHECK_MPI(MPI_Initialized(&mpi_initialized), "MPI_Initialized", -1);
    if (!mpi_initialized) {
        // OLD: PLUGIN_FAIL(-1, "requires MPI_Init before probed launches.");
        return prepare_file_exchange_locked();
    }

    PLUGIN_CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &runtime_state.world_rank), "MPI_Comm_rank", -1);
    PLUGIN_CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &runtime_state.world_size), "MPI_Comm_size", -1);
    if (runtime_state.world_size < 1 || runtime_state.world_size > FABRICPERF_MAX_CHANNEL) {
        PLUGIN_LOG("supports 1..%d MPI ranks, got %d",
                       FABRICPERF_MAX_CHANNEL, runtime_state.world_size);
        return -1;
    }

    runtime_state.file_exchange = false;
    runtime_state.exchange_dir[0] = '\0';
    runtime_state.mpi_prepared = true;
    PLUGIN_LOG("prepared rank=%d world=%d",
               runtime_state.world_rank,
               runtime_state.world_size);
    return 0;
}

/*
 * Log a host setup hint for CUDA fabric allocation failures.
 * Fabric handles require NVIDIA capability device nodes in addition to a GPU
 * attribute. Example: missing /dev/nvidia-caps can make cuMemCreate return
 * CUDA_ERROR_NOT_PERMITTED or CUDA_ERROR_NOT_SUPPORTED.
 */
static void log_fabric_allocation_hint(CUresult result) {
    int code = (int) result;
    if (code != 800 && code != 801) {
        return;
    }

    struct stat caps_dir;
    if (stat("/dev/nvidia-caps", &caps_dir) != 0) {
        PLUGIN_LOG("CUDA fabric allocation failed with %d and /dev/nvidia-caps is absent; check NVIDIA IMEX/fabric capability device setup.", code);
    } else {
        PLUGIN_LOG("CUDA fabric allocation failed with %d while /dev/nvidia-caps exists; check NVIDIA IMEX/fabric permissions.", code);
    }
}

/*
 * Allocate a local CUDA fabric-exportable buffer.
 * The returned device pointer is process-local; fabric_handle is exchanged via
 * MPI so peer ranks can import and map the same allocation.
 */
static int create_fabric_buffer(size_t requested_size, CUdevice cu_dev,
                                fabricperf_slot_t** ptr, size_t* mapped_size,
                                CUmemGenericAllocationHandle* handle,
                                CUmemFabricHandle* fabric_handle) {
    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = cu_dev;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

    /* CUDA VMM requires allocation sizes aligned to reported granularity. */
    size_t granularity = 0;
    if (PLUGIN_CHECK_CUDA(
            cuda_driver.pfn_cuMemGetAllocationGranularity(
                &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
            "cuMemGetAllocationGranularity") != 0) {
        return -1;
    }
    /* Round up to CUDA-reported granularity so VMM allocation requirements are met. */
    *mapped_size = ((requested_size + granularity - 1) / granularity) * granularity;

    /* Create physical allocation, reserve VA, and map allocation into it. */
    CUresult create_result = cuda_driver.pfn_cuMemCreate(handle, *mapped_size, &prop, 0);
    if (PLUGIN_CHECK_CUDA(create_result, "cuMemCreate") != 0) {
        log_fabric_allocation_hint(create_result);
        return -1;
    }

    CUdeviceptr addr = 0;
    if (PLUGIN_CHECK_CUDA(
            cuda_driver.pfn_cuMemAddressReserve(&addr, *mapped_size, 0, 0, 0),
            "cuMemAddressReserve") != 0 ||
        PLUGIN_CHECK_CUDA(
            cuda_driver.pfn_cuMemMap(addr, *mapped_size, 0, *handle, 0),
            "cuMemMap") != 0) {
        return -1;
    }

    CUmemAccessDesc access;
    memset(&access, 0, sizeof(access));
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = cu_dev;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    /* Make the mapping usable by this device and export it for peer import. */
    if (PLUGIN_CHECK_CUDA(cuda_driver.pfn_cuMemSetAccess(addr, *mapped_size, &access, 1),
                                  "cuMemSetAccess") != 0 ||
        PLUGIN_CHECK_CUDA(
            cuda_driver.pfn_cuMemExportToShareableHandle(
                fabric_handle, *handle, CU_MEM_HANDLE_TYPE_FABRIC, 0),
            "cuMemExportToShareableHandle") != 0) {
        return -1;
    }

    *ptr = (fabricperf_slot_t*) (uintptr_t) addr;
    return PLUGIN_CHECK_CUDA(
        cuda_memset_d8(addr, 0, *mapped_size),
        "cuMemsetD8");
}

/* Import a peer rank's fabric handle and map it into this rank's VA space. */
static int import_fabric_buffer(const CUmemFabricHandle* fabric_handle,
                                size_t mapped_size, CUdevice cu_dev,
                                fabricperf_slot_t** ptr,
                                CUmemGenericAllocationHandle* handle) {
    if (PLUGIN_CHECK_CUDA(
            cuda_driver.pfn_cuMemImportFromShareableHandle(
                handle, (void*) fabric_handle, CU_MEM_HANDLE_TYPE_FABRIC),
            "cuMemImportFromShareableHandle") != 0) {
        return -1;
    }

    CUdeviceptr addr = 0;
    if (PLUGIN_CHECK_CUDA(
            cuda_driver.pfn_cuMemAddressReserve(&addr, mapped_size, 0, 0, 0),
            "cuMemAddressReserve") != 0 ||
        PLUGIN_CHECK_CUDA(cuda_driver.pfn_cuMemMap(addr, mapped_size, 0, *handle, 0),
                                  "cuMemMap") != 0) {
        return -1;
    }

    CUmemAccessDesc access;
    memset(&access, 0, sizeof(access));
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = cu_dev;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (PLUGIN_CHECK_CUDA(
            cuda_driver.pfn_cuMemSetAccess(addr, mapped_size, &access, 1),
            "cuMemSetAccess") != 0) {
        return -1;
    }

    *ptr = (fabricperf_slot_t*) (uintptr_t) addr;
    return 0;
}

/* Build one rank/kind file path for non-MPI CUDA fabric-handle exchange. */
static int fabric_handle_path(char* output, size_t output_size,
                              const char* kind, int rank) {
    int written = snprintf(output, output_size, "%s/%s_rank%d.bin",
                           runtime_state.exchange_dir, kind, rank);
    if (written < 0 || (size_t) written >= output_size) {
        PLUGIN_LOG("fabric handle path is too long for %s rank %d.", kind, rank);
        return -1;
    }
    return 0;
}

/* Atomically publish one local CUDA fabric handle for file-exchange mode. */
static int write_fabric_handle_file(const char* kind,
                                    const CUmemFabricHandle* handle) {
    char path[PATH_MAX];
    char tmp_path[PATH_MAX];
    if (fabric_handle_path(path, sizeof(path), kind, runtime_state.world_rank) != 0) {
        return -1;
    }
    int written = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%ld",
                           path, (long) getpid());
    if (written < 0 || (size_t) written >= sizeof(tmp_path)) {
        PLUGIN_LOG("temporary fabric handle path is too long for %s.", kind);
        return -1;
    }

    FILE* file = fopen(tmp_path, "wb");
    if (file == NULL) {
        PLUGIN_LOG("could not open %s for fabric handle write.", tmp_path);
        return -1;
    }
    size_t count = fwrite(handle, sizeof(*handle), 1, file);
    if (fclose(file) != 0 || count != 1) {
        PLUGIN_LOG("could not write %s fabric handle.", kind);
        unlink(tmp_path);
        return -1;
    }
    if (rename(tmp_path, path) != 0) {
        PLUGIN_LOG("could not publish %s fabric handle at %s.", kind, path);
        unlink(tmp_path);
        return -1;
    }
    return 0;
}

/* Read one peer CUDA fabric handle if its file has been fully published. */
static int read_fabric_handle_file(const char* kind, int rank,
                                   CUmemFabricHandle* handle) {
    char path[PATH_MAX];
    if (fabric_handle_path(path, sizeof(path), kind, rank) != 0) {
        return -1;
    }
    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    size_t count = fread(handle, sizeof(*handle), 1, file);
    int close_status = fclose(file);
    if (count != 1 || close_status != 0) {
        return 1;
    }
    return 0;
}

/* Exchange all rank fabric handles without MPI using a shared filesystem. */
static int exchange_fabric_handles_via_files(CUmemFabricHandle* leader_handles,
                                             CUmemFabricHandle* follower_handles,
                                             CUmemFabricHandle* mailbox_handles,
                                             const CUmemFabricHandle* local_leader_handle,
                                             const CUmemFabricHandle* local_follower_handle,
                                             const CUmemFabricHandle* local_mailbox_handle) {
    if (write_fabric_handle_file("leader", local_leader_handle) != 0 ||
        write_fabric_handle_file("follower", local_follower_handle) != 0 ||
        write_fabric_handle_file("mailbox", local_mailbox_handle) != 0) {
        return -1;
    }

    const char* const timeout_envs[] = {
        FABRICPERF_EXCHANGE_TIMEOUT_MS_ENV,
        NULL,
    };
    int timeout_ms = selected_env_int(timeout_envs, FABRICPERF_EXCHANGE_TIMEOUT_MS_DEFAULT);
    if (timeout_ms < 1) {
        timeout_ms = FABRICPERF_EXCHANGE_TIMEOUT_MS_DEFAULT;
    }
    const int sleep_us = 10000;
    int waited_ms = 0;
    while (waited_ms <= timeout_ms) {
        int ready = 1;
        for (int rank = 0; rank < runtime_state.world_size; rank++) {
            int leader_ready = read_fabric_handle_file("leader", rank, &leader_handles[rank]);
            int follower_ready = read_fabric_handle_file("follower", rank, &follower_handles[rank]);
            int mailbox_ready = read_fabric_handle_file("mailbox", rank, &mailbox_handles[rank]);
            if (leader_ready < 0 || follower_ready < 0 || mailbox_ready < 0) {
                return -1;
            }
            if (leader_ready != 0 || follower_ready != 0 || mailbox_ready != 0) {
                ready = 0;
                break;
            }
        }
        if (ready) {
            return 0;
        }
        usleep((useconds_t) sleep_us);
        waited_ms += sleep_us / 1000;
    }
    PLUGIN_LOG("timed out waiting %d ms for non-MPI fabric handles in %s.",
               timeout_ms,
               runtime_state.exchange_dir);
    return -1;
}

/* Unmap and release virtual address space for a local or imported fabric view. */
static void unmap_fabric_buffer(fabricperf_slot_t* ptr, size_t mapped_size) {
    if (ptr == NULL) {
        return;
    }
    CUdeviceptr addr = (CUdeviceptr) (uintptr_t) ptr;
    if (cuda_driver.pfn_cuMemUnmap != NULL) {
        cuda_driver.pfn_cuMemUnmap(addr, mapped_size);
    }
    if (cuda_driver.pfn_cuMemAddressFree != NULL) {
        cuda_driver.pfn_cuMemAddressFree(addr, mapped_size);
    }
}

/*
 * Lazily allocate all FabricPerf device resources.
 * This routine exchanges fabric handles with MPI, imports peer buffers, and
 * publishes device-side pointer tables that probe PTX can load from symbols.
 */
static int prepare_buffers_locked(void) {
    if (runtime_state.buffers_prepared) {
        return 0;
    }
    if (prepare_mpi_locked() != 0) {
        return -1;
    }

    if (PLUGIN_CHECK_CUDA(
            cuda_driver.pfn_cuCtxGetDevice(&runtime_state.device),
            "cuCtxGetDevice") != 0) {
        return -1;
    }

    int vmm_supported = 0;
    int fabric_supported = 0;
    /* Fail before allocation if the selected device cannot support fabric VMM. */
    if (PLUGIN_CHECK_CUDA(
            cuda_driver.pfn_cuDeviceGetAttribute(
                &vmm_supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                runtime_state.device),
            "cuDeviceGetAttribute(VMM)") != 0 ||
        PLUGIN_CHECK_CUDA(
            cuda_driver.pfn_cuDeviceGetAttribute(
                &fabric_supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                runtime_state.device),
            "cuDeviceGetAttribute(FABRIC)") != 0) {
        return -1;
    }
    if (!vmm_supported || !fabric_supported) {
        PLUGIN_LOG("requires CUDA VMM and fabric-handle support (rank=%d vmm=%d fabric=%d)",
                       runtime_state.world_rank, vmm_supported, fabric_supported);
        return -1;
    }

    runtime_state.leader_handles = (CUmemGenericAllocationHandle*)
        calloc((size_t) runtime_state.world_size, sizeof(CUmemGenericAllocationHandle));
    runtime_state.follower_handles = (CUmemGenericAllocationHandle*)
        calloc((size_t) runtime_state.world_size, sizeof(CUmemGenericAllocationHandle));
    runtime_state.mailbox_handles = (CUmemGenericAllocationHandle*)
        calloc((size_t) runtime_state.world_size, sizeof(CUmemGenericAllocationHandle));
    runtime_state.leader_buffs = (fabricperf_slot_t**)
        calloc((size_t) runtime_state.world_size, sizeof(fabricperf_slot_t*));
    runtime_state.follower_buffs = (fabricperf_slot_t**)
        calloc((size_t) runtime_state.world_size, sizeof(fabricperf_slot_t*));
    runtime_state.mailbox_buffs = (fabricperf_slot_t**)
        calloc((size_t) runtime_state.world_size, sizeof(fabricperf_slot_t*));
    if (runtime_state.leader_handles == NULL || runtime_state.follower_handles == NULL ||
        runtime_state.mailbox_handles == NULL || runtime_state.leader_buffs == NULL ||
        runtime_state.follower_buffs == NULL || runtime_state.mailbox_buffs == NULL) {
        PLUGIN_FAIL(-1, "failed to allocate host bookkeeping arrays.");
    }

    /* Keep sizing math in one place so symbol writes only publish pointers. */
    const size_t ptp_sequence_slots =
        FABRICPERF_PTP_SAMPLES_PER_LEADER * (size_t) runtime_state.world_size;
    const size_t sequence_slots =
        ptp_sequence_slots + FABRICPERF_SENDRECV_MESSAGE_SLOTS;
    const size_t leader_requested_size =
        sequence_slots * (size_t) runtime_state.world_size *
        FABRICPERF_LEADER_SPACES * sizeof(fabricperf_slot_t);
    const size_t follower_requested_size =
        sequence_slots * FABRICPERF_FOLLOWER_SPACES * sizeof(fabricperf_slot_t);
    const size_t mailbox_requested_size =
        (size_t) runtime_state.world_size * FABRICPERF_LATENCY_MAILBOX_VBLOCKS *
        sizeof(fabricperf_slot_t);

    CUmemFabricHandle local_leader_handle;
    CUmemFabricHandle local_follower_handle;
    CUmemFabricHandle local_mailbox_handle;
    /* Create this rank's exportable buffers before exchanging handles. */
    if (create_fabric_buffer(leader_requested_size, runtime_state.device,
                             &runtime_state.local_leader_buff,
                             &runtime_state.leader_alloc_size,
                             &runtime_state.leader_handles[runtime_state.world_rank],
                             &local_leader_handle) != 0 ||
        create_fabric_buffer(follower_requested_size, runtime_state.device,
                             &runtime_state.local_follower_buff,
                             &runtime_state.follower_alloc_size,
                             &runtime_state.follower_handles[runtime_state.world_rank],
                             &local_follower_handle) != 0 ||
        create_fabric_buffer(mailbox_requested_size, runtime_state.device,
                             &runtime_state.local_mailbox_buff,
                             &runtime_state.mailbox_alloc_size,
                             &runtime_state.mailbox_handles[runtime_state.world_rank],
                             &local_mailbox_handle) != 0) {
        return -1;
    }

    CUmemFabricHandle* leader_handles = (CUmemFabricHandle*)
        malloc((size_t) runtime_state.world_size * sizeof(CUmemFabricHandle));
    CUmemFabricHandle* follower_handles = (CUmemFabricHandle*)
        malloc((size_t) runtime_state.world_size * sizeof(CUmemFabricHandle));
    CUmemFabricHandle* mailbox_handles = (CUmemFabricHandle*)
        malloc((size_t) runtime_state.world_size * sizeof(CUmemFabricHandle));
    if (leader_handles == NULL || follower_handles == NULL || mailbox_handles == NULL) {
        free(leader_handles);
        free(follower_handles);
        free(mailbox_handles);
        PLUGIN_FAIL(-1, "failed to allocate MPI handle buffers.");
    }

    /* MPI counts are int; keep the cast explicit and checked. */
    if (sizeof(CUmemFabricHandle) > (size_t) INT_MAX) {
        free(leader_handles);
        free(follower_handles);
        free(mailbox_handles);
        PLUGIN_FAIL(-1, "CUmemFabricHandle is too large for MPI_Allgather.");
    }
    if (runtime_state.file_exchange) {
        if (exchange_fabric_handles_via_files(leader_handles,
                                              follower_handles,
                                              mailbox_handles,
                                              &local_leader_handle,
                                              &local_follower_handle,
                                              &local_mailbox_handle) != 0) {
            free(leader_handles);
            free(follower_handles);
            free(mailbox_handles);
            return -1;
        }
    } else {
        /* Exchange opaque CUDA fabric handles directly with MPI_BYTE. */
        int mpi_status = MPI_Allgather(&local_leader_handle, (int) sizeof(CUmemFabricHandle), MPI_BYTE,
                                       leader_handles, (int) sizeof(CUmemFabricHandle), MPI_BYTE,
                                       MPI_COMM_WORLD);
        if (mpi_status != MPI_SUCCESS) {
            PLUGIN_LOG_MPI_ERROR(mpi_status, "MPI_Allgather(leader)");
            free(leader_handles);
            free(follower_handles);
            free(mailbox_handles);
            return -1;
        }
        mpi_status = MPI_Allgather(&local_follower_handle, (int) sizeof(CUmemFabricHandle), MPI_BYTE,
                                   follower_handles, (int) sizeof(CUmemFabricHandle), MPI_BYTE,
                                   MPI_COMM_WORLD);
        if (mpi_status != MPI_SUCCESS) {
            PLUGIN_LOG_MPI_ERROR(mpi_status, "MPI_Allgather(follower)");
            free(leader_handles);
            free(follower_handles);
            free(mailbox_handles);
            return -1;
        }
        mpi_status = MPI_Allgather(&local_mailbox_handle, (int) sizeof(CUmemFabricHandle), MPI_BYTE,
                                   mailbox_handles, (int) sizeof(CUmemFabricHandle), MPI_BYTE,
                                   MPI_COMM_WORLD);
        if (mpi_status != MPI_SUCCESS) {
            PLUGIN_LOG_MPI_ERROR(mpi_status, "MPI_Allgather(mailbox)");
            free(leader_handles);
            free(follower_handles);
            free(mailbox_handles);
            return -1;
        }
    }

    for (int idx = 0; idx < runtime_state.world_size; idx++) {
        if (idx == runtime_state.world_rank) {
            /* Local rank already owns mapped pointers from create_fabric_buffer. */
            runtime_state.leader_buffs[idx] = runtime_state.local_leader_buff;
            runtime_state.follower_buffs[idx] = runtime_state.local_follower_buff;
            runtime_state.mailbox_buffs[idx] = runtime_state.local_mailbox_buff;
        } else if (import_fabric_buffer(&leader_handles[idx],
                                        runtime_state.leader_alloc_size,
                                        runtime_state.device,
                                        &runtime_state.leader_buffs[idx],
                                        &runtime_state.leader_handles[idx]) != 0 ||
                   import_fabric_buffer(&follower_handles[idx],
                                        runtime_state.follower_alloc_size,
                                        runtime_state.device,
                                        &runtime_state.follower_buffs[idx],
                                        &runtime_state.follower_handles[idx]) != 0 ||
                   import_fabric_buffer(&mailbox_handles[idx],
                                        runtime_state.mailbox_alloc_size,
                                        runtime_state.device,
                                        &runtime_state.mailbox_buffs[idx],
                                        &runtime_state.mailbox_handles[idx]) != 0) {
            free(leader_handles);
            free(follower_handles);
            free(mailbox_handles);
            return -1;
        }
    }
    free(leader_handles);
    free(follower_handles);
    free(mailbox_handles);

    /* Copy host pointer tables into device memory for probe-side indirection. */
    if (PLUGIN_CHECK_CUDA(
            cuda_driver.pfn_cuMemAlloc_v2(
                &runtime_state.device_leader_table,
                (size_t) runtime_state.world_size * sizeof(fabricperf_slot_t*)),
            "cuMemAlloc(leader_table)") != 0 ||
        PLUGIN_CHECK_CUDA(
            cuda_memcpy_htod(runtime_state.device_leader_table,
                             runtime_state.leader_buffs,
                             (size_t) runtime_state.world_size * sizeof(fabricperf_slot_t*)),
            "cuMemcpyHtoD(leader_table)") != 0 ||
        PLUGIN_CHECK_CUDA(
            cuda_driver.pfn_cuMemAlloc_v2(
                &runtime_state.device_follower_table,
                (size_t) runtime_state.world_size * sizeof(fabricperf_slot_t*)),
            "cuMemAlloc(follower_table)") != 0 ||
        PLUGIN_CHECK_CUDA(
            cuda_memcpy_htod(runtime_state.device_follower_table,
                             runtime_state.follower_buffs,
                             (size_t) runtime_state.world_size * sizeof(fabricperf_slot_t*)),
            "cuMemcpyHtoD(follower_table)") != 0 ||
        PLUGIN_CHECK_CUDA(
            cuda_driver.pfn_cuMemAlloc_v2(
                &runtime_state.device_mailbox_table,
                (size_t) runtime_state.world_size * sizeof(fabricperf_slot_t*)),
            "cuMemAlloc(mailbox_table)") != 0 ||
        PLUGIN_CHECK_CUDA(
            cuda_memcpy_htod(runtime_state.device_mailbox_table,
                             runtime_state.mailbox_buffs,
                             (size_t) runtime_state.world_size * sizeof(fabricperf_slot_t*)),
            "cuMemcpyHtoD(mailbox_table)") != 0) {
        return -1;
    }

    /* A shared result scratch is reset before each result-symbol launch. */
    runtime_state.result_buffer_size = FABRICPERF_RESULT_SLOTS * sizeof(fabricperf_slot_t);
    if (PLUGIN_CHECK_CUDA(
            cuda_driver.pfn_cuMemAlloc_v2(&runtime_state.device_result_buffer,
                                         runtime_state.result_buffer_size),
            "cuMemAlloc(result_buffer)") != 0 ||
        PLUGIN_CHECK_CUDA(
            cuda_memset_d8(runtime_state.device_result_buffer, 0,
                           runtime_state.result_buffer_size),
            "cuMemsetD8(result_buffer)") != 0) {
        return -1;
    }

    /*
     * Allocate a local offset table indexed by srcPeer.
     * Example: offset[src] is added to a timestamp written by rank src so the
     * receive hook compares it against this rank's local %globaltimer domain.
     */
    runtime_state.latency_offset_buffer_size =
        (size_t) runtime_state.world_size * sizeof(uint64_t);
    if (PLUGIN_CHECK_CUDA(
            cuda_driver.pfn_cuMemAlloc_v2(&runtime_state.device_latency_offset_buffer,
                                         runtime_state.latency_offset_buffer_size),
            "cuMemAlloc(latency_offset_buffer)") != 0 ||
        PLUGIN_CHECK_CUDA(
            cuda_memset_d8(runtime_state.device_latency_offset_buffer, 0,
                           runtime_state.latency_offset_buffer_size),
            "cuMemsetD8(latency_offset_buffer)") != 0) {
        return -1;
    }

    runtime_state.buffers_prepared = true;
    if (!runtime_state.reported) {
        /* Emit one preparation summary per process for offline trace diagnostics. */
        PLUGIN_LOG("prepared rank=%d world=%d device=%d",
                       runtime_state.world_rank, runtime_state.world_size, runtime_state.device);
        runtime_state.reported = true;
    }
    return 0;
}

/* Required Neutrino plugin entry point: validate ABI and resolve libcuda. */
int neutrino_plugin_init_v1(const neutrino_plugin_api_v1* plugin_api) {
    api = plugin_api;
    PLUGIN_REQUIRE_API(-1);
    if (load_cuda_driver() != 0) {
        return -1;
    }
    /* Initialization diagnostics are cold-path only and safe to route through the API logger. */
    PLUGIN_LOG("initialized");
    return 0;
}

/* Optional Neutrino plugin cleanup hook. */
void neutrino_plugin_fini_v1(void) {
    pthread_mutex_lock(&runtime_mutex);
    /* Synchronize first so no kernel still touches buffers being unmapped. */
    if (runtime_state.buffers_prepared && cuda_driver.pfn_cuCtxSynchronize != NULL) {
        cuda_driver.pfn_cuCtxSynchronize();
    }

    /* Free simple device allocations before releasing VMM/fabric mappings. */
    if (runtime_state.device_leader_table != 0 && cuda_driver.pfn_cuMemFree_v2 != NULL) {
        cuda_driver.pfn_cuMemFree_v2(runtime_state.device_leader_table);
    }
    if (runtime_state.device_follower_table != 0 && cuda_driver.pfn_cuMemFree_v2 != NULL) {
        cuda_driver.pfn_cuMemFree_v2(runtime_state.device_follower_table);
    }
    if (runtime_state.device_mailbox_table != 0 && cuda_driver.pfn_cuMemFree_v2 != NULL) {
        cuda_driver.pfn_cuMemFree_v2(runtime_state.device_mailbox_table);
    }
    if (runtime_state.device_result_buffer != 0 && cuda_driver.pfn_cuMemFree_v2 != NULL) {
        cuda_driver.pfn_cuMemFree_v2(runtime_state.device_result_buffer);
    }
    if (runtime_state.device_latency_offset_buffer != 0 && cuda_driver.pfn_cuMemFree_v2 != NULL) {
        cuda_driver.pfn_cuMemFree_v2(runtime_state.device_latency_offset_buffer);
    }

    if (runtime_state.leader_buffs != NULL) {
        /* Unmap all local views, including imported peer views. */
        for (int idx = 0; idx < runtime_state.world_size; idx++) {
            unmap_fabric_buffer(runtime_state.leader_buffs[idx], runtime_state.leader_alloc_size);
        }
        free(runtime_state.leader_buffs);
    }
    if (runtime_state.follower_buffs != NULL) {
        for (int idx = 0; idx < runtime_state.world_size; idx++) {
            unmap_fabric_buffer(runtime_state.follower_buffs[idx], runtime_state.follower_alloc_size);
        }
        free(runtime_state.follower_buffs);
    }
    if (runtime_state.mailbox_buffs != NULL) {
        for (int idx = 0; idx < runtime_state.world_size; idx++) {
            unmap_fabric_buffer(runtime_state.mailbox_buffs[idx], runtime_state.mailbox_alloc_size);
        }
        free(runtime_state.mailbox_buffs);
    }

    if (runtime_state.leader_handles != NULL) {
        /* Release VMM handles after VA mappings have been removed. */
        for (int idx = 0; idx < runtime_state.world_size; idx++) {
            if (runtime_state.leader_handles[idx] != 0 && cuda_driver.pfn_cuMemRelease != NULL) {
                cuda_driver.pfn_cuMemRelease(runtime_state.leader_handles[idx]);
            }
        }
        free(runtime_state.leader_handles);
    }
    if (runtime_state.follower_handles != NULL) {
        for (int idx = 0; idx < runtime_state.world_size; idx++) {
            if (runtime_state.follower_handles[idx] != 0 && cuda_driver.pfn_cuMemRelease != NULL) {
                cuda_driver.pfn_cuMemRelease(runtime_state.follower_handles[idx]);
            }
        }
        free(runtime_state.follower_handles);
    }
    if (runtime_state.mailbox_handles != NULL) {
        for (int idx = 0; idx < runtime_state.world_size; idx++) {
            if (runtime_state.mailbox_handles[idx] != 0 && cuda_driver.pfn_cuMemRelease != NULL) {
                cuda_driver.pfn_cuMemRelease(runtime_state.mailbox_handles[idx]);
            }
        }
        free(runtime_state.mailbox_handles);
    }

    memset(&runtime_state, 0, sizeof(runtime_state));
    if (cuda_driver.handle != NULL) {
        dlclose(cuda_driver.handle);
        memset(&cuda_driver, 0, sizeof(cuda_driver));
    }
    pthread_mutex_unlock(&runtime_mutex);
}

/*
 * Optional prepare hook.
 * Scan requested symbols so FabricPerf only pays MPI/VMM setup cost for probes
 * that actually require rank IDs or fabric pointers.
 */
int neutrino_plugin_prepare_launch_v1(const char* kernel_name,
                                      const neutrino_plugin_symbol_v1* symbols,
                                      int n_symbols,
                                      unsigned int launch_index,
                                      const neutrino_plugin_module_context_v1* context) {
    (void) kernel_name;
    (void) launch_index;
    (void) context;
    bool needs_mpi = false;
    bool needs_buffers = false;

    for (int idx = 0; idx < n_symbols; idx++) {
        /* MPI is needed for rank/size symbols and also for buffer-backed symbols. */
        needs_mpi = needs_mpi ||
                    PLUGIN_IS_SYMBOL(symbols[idx].name, "deviceId") != 0 ||
                    PLUGIN_IS_SYMBOL(symbols[idx].name, "numDevices") != 0 ||
                    PLUGIN_IS_SYMBOL(symbols[idx].name, "globalLeaderBuff") != 0 ||
                    PLUGIN_IS_SYMBOL(symbols[idx].name, "globalFollowerBuff") != 0 ||
                    PLUGIN_IS_SYMBOL(symbols[idx].name, "globalLatencyMailboxBuff") != 0 ||
                    PLUGIN_IS_SYMBOL(symbols[idx].name, "globalLatencyOffsetBuff") != 0 ||
                    PLUGIN_IS_SYMBOL(symbols[idx].name, "globalResultBuff") != 0 ||
                    PLUGIN_IS_SYMBOL(symbols[idx].name, "latencyGlobal") != 0;

        /* Buffer-backed symbols require fabric handle exchange/mapping before launch. */
        needs_buffers = needs_buffers ||
                        PLUGIN_IS_SYMBOL(symbols[idx].name, "globalLeaderBuff") != 0 ||
                        PLUGIN_IS_SYMBOL(symbols[idx].name, "globalFollowerBuff") != 0 ||
                        PLUGIN_IS_SYMBOL(symbols[idx].name, "globalLatencyMailboxBuff") != 0 ||
                        PLUGIN_IS_SYMBOL(symbols[idx].name, "globalLatencyOffsetBuff") != 0 ||
                        PLUGIN_IS_SYMBOL(symbols[idx].name, "globalResultBuff") != 0 ||
                        PLUGIN_IS_SYMBOL(symbols[idx].name, "latencyGlobal") != 0;
    }

    if (needs_buffers) {
        int status;
        /* Serialize lazy MPI/VMM setup because the runtime state is process-global. */
        pthread_mutex_lock(&runtime_mutex);
        status = prepare_buffers_locked();
        pthread_mutex_unlock(&runtime_mutex);
        return status;
    }
    if (needs_mpi) {
        int status;
        /* Serialize lazy MPI discovery for the same shared runtime state. */
        pthread_mutex_lock(&runtime_mutex);
        status = prepare_mpi_locked();
        pthread_mutex_unlock(&runtime_mutex);
        return status;
    }
    return 0;
}

/*
 * Optional symbol resolver.
 * Return 1 when FabricPerf writes a symbol, 0 when Neutrino should continue
 * with normal copy/fallback handling, and -1 on hard plugin failure.
 */
int neutrino_plugin_resolve_symbol_v1(const neutrino_plugin_symbol_v1* symbol,
                                      void* destination_module,
                                      const neutrino_plugin_module_context_v1* context) {
    if (symbol == NULL) {
        return 0;
    }
    CUmodule destination = (CUmodule) destination_module;
    const char* symbol_name = symbol->name;
    if (PLUGIN_IS_SYMBOL(symbol_name, "deviceId") != 0) {
        /* Device ID is MPI rank for FabricPerf's cross-rank protocol. */
        int status;
        pthread_mutex_lock(&runtime_mutex);
        status = prepare_mpi_locked();
        pthread_mutex_unlock(&runtime_mutex);
        if (status != 0) return -1;
        uint32_t value = (uint32_t) runtime_state.world_rank;
        return symbol_write(destination, symbol_name, &value, sizeof(value));
    }
    if (PLUGIN_IS_SYMBOL(symbol_name, "numDevices") != 0) {
        /* numDevices is MPI world size, not CUDA device count, in FabricPerf. */
        int status;
        pthread_mutex_lock(&runtime_mutex);
        status = prepare_mpi_locked();
        pthread_mutex_unlock(&runtime_mutex);
        if (status != 0) return -1;
        uint32_t value = (uint32_t) runtime_state.world_size;
        return symbol_write(destination, symbol_name, &value, sizeof(value));
    }
    if (PLUGIN_IS_SYMBOL(symbol_name, "hostId") != 0) {
        /* Keep hostId FabricPerf-scoped; example: FABRICPERF_HOSTID=2 writes hostId=2. */
        const char* raw = getenv("FABRICPERF_HOSTID");
        uint32_t value = raw != NULL ? (uint32_t) atoi(raw) : 0u;
        return symbol_write(destination, symbol_name, &value, sizeof(value));
    }
    if (PLUGIN_IS_SYMBOL(symbol_name, "ptpRunId") != 0) {
        /* Launch index comes from Neutrino context, not plugin-local state. */
        uint32_t value = context != NULL ? context->launch_index : 0u;
        return symbol_write(destination, symbol_name, &value, sizeof(value));
    }
    if (PLUGIN_IS_SYMBOL(symbol_name, "ptpGlobalBarrier") != 0 ||
        PLUGIN_IS_SYMBOL(symbol_name, "ptpGlobalBarrierSense") != 0) {
        /* Grid barrier globals must begin at zero for each generated module. */
        uint32_t value = 0u;
        return symbol_write(destination, symbol_name, &value, sizeof(value));
    }
    if (PLUGIN_IS_SYMBOL(symbol_name, "globalFollowerBuff") != 0) {
        int status;
        pthread_mutex_lock(&runtime_mutex);
        status = prepare_buffers_locked();
        pthread_mutex_unlock(&runtime_mutex);
        if (status != 0) return -1;
        return symbol_write(destination,
                            symbol_name,
                            &runtime_state.device_follower_table,
                            sizeof(runtime_state.device_follower_table));
    }
    if (PLUGIN_IS_SYMBOL(symbol_name, "globalLeaderBuff") != 0) {
        int status;
        pthread_mutex_lock(&runtime_mutex);
        status = prepare_buffers_locked();
        pthread_mutex_unlock(&runtime_mutex);
        if (status != 0) return -1;
        return symbol_write(destination,
                            symbol_name,
                            &runtime_state.device_leader_table,
                            sizeof(runtime_state.device_leader_table));
    }
    if (PLUGIN_IS_SYMBOL(symbol_name, "globalLatencyMailboxBuff") != 0) {
        int status;
        pthread_mutex_lock(&runtime_mutex);
        status = prepare_buffers_locked();
        pthread_mutex_unlock(&runtime_mutex);
        if (status != 0) return -1;
        /*
         * OLD: clearing the rank-owned mailbox here could race peers on scales
         * without the entry PTP barrier. The PTX tag includes ptpRunId instead.
         */
        /* if (cuda_memset_d8((CUdeviceptr) (uintptr_t) runtime_state.local_mailbox_buff, 0,
                              runtime_state.mailbox_alloc_size) != CUDA_SUCCESS) {
            return -1;
        } */
        return symbol_write(destination,
                            symbol_name,
                            &runtime_state.device_mailbox_table,
                            sizeof(runtime_state.device_mailbox_table));
    }
    if (PLUGIN_IS_SYMBOL(symbol_name, "globalLatencyOffsetBuff") != 0) {
        int status;
        pthread_mutex_lock(&runtime_mutex);
        status = prepare_buffers_locked();
        pthread_mutex_unlock(&runtime_mutex);
        if (status != 0) return -1;
        /*
         * PTP currently calibrates root-relative offsets only through 8 ranks.
         * Example: at 16 ranks the send/receive hooks use zero offsets rather
         * than reusing stale values from a previous launch.
         */
        if (runtime_state.world_size > FABRICPERF_MAX_PTP_RANKS &&
            !runtime_state.ptp_scale_reported) {
            PLUGIN_LOG("latency PTP supports 2..%d ranks; got %d, using zero clock offsets.",
                       FABRICPERF_MAX_PTP_RANKS, runtime_state.world_size);
            runtime_state.ptp_scale_reported = true;
        }
        if (cuda_memset_d8(runtime_state.device_latency_offset_buffer, 0,
                           runtime_state.latency_offset_buffer_size) != CUDA_SUCCESS) {
            return -1;
        }
        return symbol_write(destination,
                            symbol_name,
                            &runtime_state.device_latency_offset_buffer,
                            sizeof(runtime_state.device_latency_offset_buffer));
    }
    if (PLUGIN_IS_SYMBOL(symbol_name, "globalResultBuff") != 0 ||
        PLUGIN_IS_SYMBOL(symbol_name, "latencyGlobal") != 0) {
        int status;
        pthread_mutex_lock(&runtime_mutex);
        status = prepare_buffers_locked();
        pthread_mutex_unlock(&runtime_mutex);
        if (status != 0) return -1;
        /* Result symbols reuse one scratch buffer; clear it before publishing. */
        if (cuda_memset_d8(runtime_state.device_result_buffer, 0,
                           runtime_state.result_buffer_size) != CUDA_SUCCESS) {
            return -1;
        }
        return symbol_write(destination,
                            symbol_name,
                            &runtime_state.device_result_buffer,
                            sizeof(runtime_state.device_result_buffer));
    }
    return 0;
}
