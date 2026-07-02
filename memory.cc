#include "common.h"

#include <cupti_pmsampling.h>
#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_result.h>
#include <cupti_target.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iomanip>
#include <mpi.h>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

static const neutrino_plugin_api_v1* api = nullptr;

struct DriverApi {
    void* handle = nullptr;
    CUresult (*pfn_cuCtxSynchronize)(void) = nullptr;
    CUresult (*pfn_cuCtxGetCurrent)(CUcontext*) = nullptr;
    CUresult (*pfn_cuCtxGetDevice)(CUdevice*) = nullptr;
    CUresult (*pfn_cuDeviceGetAttribute)(int*, CUdevice_attribute, CUdevice) = nullptr;
    CUresult (*pfn_cuStreamSynchronize)(CUstream) = nullptr;
    CUresult (*pfn_cuEventCreate)(CUevent*, unsigned int) = nullptr;
    CUresult (*pfn_cuEventRecord)(CUevent, CUstream) = nullptr;
    CUresult (*pfn_cuEventSynchronize)(CUevent) = nullptr;
    CUresult (*pfn_cuEventElapsedTime)(float*, CUevent, CUevent) = nullptr;
    CUresult (*pfn_cuEventDestroy)(CUevent) = nullptr;
    CUresult (*pfn_cuEventDestroy_v2)(CUevent) = nullptr;
    CUresult (*pfn_cuMemGetAllocationGranularity)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags) = nullptr;
    CUresult (*pfn_cuMemCreate)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long) = nullptr;
    CUresult (*pfn_cuMemAddressReserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long) = nullptr;
    CUresult (*pfn_cuMemMap)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long) = nullptr;
    CUresult (*pfn_cuMemSetAccess)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) = nullptr;
    CUresult (*pfn_cuMemExportToShareableHandle)(void*, CUmemGenericAllocationHandle, CUmemAllocationHandleType, unsigned long long) = nullptr;
    CUresult (*pfn_cuMemImportFromShareableHandle)(CUmemGenericAllocationHandle*, void*, CUmemAllocationHandleType) = nullptr;
    CUresult (*pfn_cuMemUnmap)(CUdeviceptr, size_t) = nullptr;
    CUresult (*pfn_cuMemAddressFree)(CUdeviceptr, size_t) = nullptr;
    CUresult (*pfn_cuMemRelease)(CUmemGenericAllocationHandle) = nullptr;
    CUresult (*pfn_cuMemAlloc_v2)(CUdeviceptr*, size_t) = nullptr;
    CUresult (*pfn_cuMemFree_v2)(CUdeviceptr) = nullptr;
    CUresult (*pfn_cuMemsetD8)(CUdeviceptr, unsigned char, size_t) = nullptr;
    CUresult (*pfn_cuMemsetD8_v2)(CUdeviceptr, unsigned char, size_t) = nullptr;
    CUresult (*pfn_cuMemcpyHtoD)(CUdeviceptr, const void*, size_t) = nullptr;
    CUresult (*pfn_cuMemcpyHtoD_v2)(CUdeviceptr, const void*, size_t) = nullptr;
    CUresult (*pfn_cuModuleGetGlobal)(CUdeviceptr*, size_t*, CUmodule, const char*) = nullptr;
    CUresult (*pfn_cuModuleGetGlobal_v2)(CUdeviceptr*, size_t*, CUmodule, const char*) = nullptr;
};

struct FabricPerfRuntime {
    bool mpi_prepared = false; // Records whether rank/size were read; example: symbol deviceId forces MPI setup.
    bool buffers_prepared = false; // Records whether cross-rank VMM buffers exist; example: globalLeaderBuff requires this.
    bool reported = false; // Emits one preparation log line per process; example: FabricPerf prepared rank=0.
    int world_rank = 0; // MPI rank used by FabricPerf's PTX protocol; example: rank 1 writes deviceId=1.
    int world_size = 0; // MPI world size used for peer table dimensions; example: two ranks allocate two table slots.
    CUdevice device = 0; // CUDA device associated with the active context; example: VMM allocations use this device id.
    size_t leader_alloc_size = 0; // Aligned byte size for leader buffers; example: CUDA VMM granularity rounds this up.
    size_t follower_alloc_size = 0; // Aligned byte size for follower buffers; example: imported peer views use this size.
    CUmemGenericAllocationHandle* leader_handles = nullptr; // Per-rank VMM handles; example: index world_rank is local.
    CUmemGenericAllocationHandle* follower_handles = nullptr; // Per-rank follower VMM handles; example: peer handles are imported.
    FabricPerfSlot** leader_buffs = nullptr; // Host table copied to device memory; example: probe loads peer leader pointer.
    FabricPerfSlot** follower_buffs = nullptr; // Host table copied to device memory; example: send timestamp probe uses this table.
    FabricPerfSlot* local_leader_buff = nullptr; // Local mapped leader buffer; example: rank 0 maps its own allocation here.
    FabricPerfSlot* local_follower_buff = nullptr; // Local mapped follower buffer; example: rank 1 maps its own allocation here.
    CUdeviceptr device_leader_table = 0; // Device-side pointer table for globalLeaderBuff.
    CUdeviceptr device_follower_table = 0; // Device-side pointer table for globalFollowerBuff.
    CUdeviceptr device_result_buffer = 0; // Device scratch buffer for globalResultBuff/latencyGlobal.
    size_t result_buffer_size = 0; // Scratch byte size; example: FABRICPERF_RESULT_SLOTS * sizeof(FabricPerfSlot).
};

struct CupTiState {
    std::mutex mutex;
    bool enabled = true;
    bool profiler_initialized = false;
    bool ready = false;
    bool active = false;
    bool active_events = false;
    int rank = 0;
    CUcontext context = nullptr;
    CUdevice device = -1;
    CUpti_PmSampling_Object* pm_sampling = nullptr;
    CUpti_Profiler_Host_Object* host_object = nullptr;
    CUevent start_event = nullptr;
    CUevent end_event = nullptr;
    FILE* csv = nullptr;
    std::string csv_path;
    std::string chip_name;
    std::vector<uint8_t> counter_availability;
    std::vector<uint8_t> config_image;
    std::vector<uint8_t> counter_data;
    std::vector<std::string> requested_metrics;
    std::vector<std::string> supported_metrics;
    std::unordered_map<std::string, CUptiResult> unsupported_metrics;
    std::unordered_map<std::string, CUptiResult> eval_errors;
    std::unordered_map<std::string, double> values;
    neutrino_plugin_launch_context_v1 launch = {};
    double event_duration_s = 0.0;
    uint64_t pm_sampling_interval = 1000;
    size_t pm_hardware_buffer_size = 8 * 1024 * 1024;
    uint32_t pm_max_samples = 256;
    CUpti_PmSampling_TriggerMode pm_trigger_mode =
        CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL;
    CUpti_PmSampling_HardwareBuffer_AppendMode pm_append_mode =
        CUPTI_PM_SAMPLING_HARDWARE_BUFFER_APPEND_MODE_KEEP_OLDEST;
    bool active_pm_sampling = false;
    bool pm_overflow = false;
    bool pm_disable_on_teardown = false;
    size_t pm_total_samples = 0;
    size_t pm_populated_samples = 0;
    size_t pm_completed_samples = 0;
};

static DriverApi driver = {};
static CupTiState state = {};
static constexpr const char* CUPTI_BACKEND_NAME = "pm_sampling"; // Stable CSV/log label for the CUPTI backend.
static std::mutex fabric_mutex; // Serializes lazy FabricPerf MPI/VMM setup; example: concurrent module loads share one buffer table.
static FabricPerfRuntime fabric = {}; // Process-local FabricPerf runtime state; example: symbol resolver publishes its device tables.

static std::string cupti_status(CUptiResult result) {
    const char* text = nullptr;
    if (cuptiGetResultString(result, &text) == CUPTI_SUCCESS && text != nullptr) {
        return text;
    }
    return std::string("CUPTI_") + std::to_string((int) result);
}

static bool truthy_value(const std::string& raw, bool fallback) {
    if (raw.empty()) {
        return fallback;
    }
    return !(raw == "0" || raw == "false" || raw == "False" ||
             raw == "FALSE" || raw == "no" || raw == "off");
}

static uint64_t parse_u64_value(const char* label,
                                const std::string& raw,
                                uint64_t fallback,
                                uint64_t minimum) {
    if (raw.empty()) {
        return fallback;
    }
    char* end = nullptr;
    unsigned long long value = std::strtoull(raw.c_str(), &end, 0);
    if (end == raw.c_str() || (end != nullptr && *end != '\0') || value < minimum) {
        PLUGIN_LOG("ignoring invalid %s=%s", label, raw.c_str());
        return fallback;
    }
    return (uint64_t) value;
}

static std::string trim(std::string value) {
    const char* ws = " \t\r\n";
    size_t begin = value.find_first_not_of(ws);
    if (begin == std::string::npos) {
        return "";
    }
    size_t end = value.find_last_not_of(ws);
    return value.substr(begin, end - begin + 1);
}

static void add_unique(std::vector<std::string>& values, const std::string& value) {
    if (value.empty()) {
        return;
    }
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

static std::vector<std::string> default_pm_metrics() {
    // Memory mode gets DRAM/XBAR bytes from the Neutrino probe; PM Sampling only supplies NVLink and duration.
    return {
        "nvlrx__bytes.sum",
        "nvltx__bytes.sum",
        "gpu__time_duration.sum",
    };
}

static std::unordered_map<std::string, std::string> parse_key_value_list(const char* raw) {
    std::unordered_map<std::string, std::string> result;
    if (raw == nullptr || raw[0] == '\0') {
        return result;
    }
    std::string current;
    for (const char* p = raw; ; ++p) {
        if (*p == ',' || *p == '\0') {
            std::string token = trim(current);
            size_t eq = token.find('=');
            if (eq != std::string::npos) {
                result[trim(token.substr(0, eq))] = trim(token.substr(eq + 1));
            } else if (!token.empty()) {
                PLUGIN_LOG("ignoring option without key=value: %s", token.c_str());
            }
            current.clear();
            if (*p == '\0') {
                break;
            }
        } else {
            current.push_back(*p);
        }
    }
    return result;
}

static void apply_pm_trigger_option(const std::string& raw) {
    if (raw == "sysclk" || raw == "cycles") {
        state.pm_trigger_mode = CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_SYSCLK_INTERVAL;
    } else if (raw == "time" || raw == "ns") {
        state.pm_trigger_mode = CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL;
    } else {
        PLUGIN_LOG("ignoring invalid dev option pm_trigger=%s", raw.c_str());
    }
}

static void apply_pm_append_option(const std::string& raw) {
    if (raw == "latest") {
        state.pm_append_mode = CUPTI_PM_SAMPLING_HARDWARE_BUFFER_APPEND_MODE_KEEP_LATEST;
    } else if (raw == "oldest") {
        state.pm_append_mode = CUPTI_PM_SAMPLING_HARDWARE_BUFFER_APPEND_MODE_KEEP_OLDEST;
    } else {
        PLUGIN_LOG("ignoring invalid dev option pm_append=%s", raw.c_str());
    }
}

static void configure_dev_options_locked() {
    std::unordered_map<std::string, std::string> options =
        parse_key_value_list(std::getenv("FABRICPERF_CUPTI_DEV_OPTIONS"));
    for (const auto& item : options) {
        const std::string& key = item.first;
        const std::string& value = item.second;
        if (key == "pm_interval_ns") {
            state.pm_sampling_interval = parse_u64_value(key.c_str(), value, state.pm_sampling_interval, 1);
        } else if (key == "pm_hardware_buffer_bytes") {
            state.pm_hardware_buffer_size =
                (size_t) parse_u64_value(key.c_str(), value, state.pm_hardware_buffer_size, 4096);
        } else if (key == "pm_max_samples") {
            state.pm_max_samples =
                (uint32_t) parse_u64_value(key.c_str(), value, state.pm_max_samples, 1);
        } else if (key == "pm_trigger") {
            apply_pm_trigger_option(value);
        } else if (key == "pm_append") {
            apply_pm_append_option(value);
        } else if (key == "pm_disable_on_teardown") {
            state.pm_disable_on_teardown = truthy_value(value, false);
        } else {
            PLUGIN_LOG("ignoring unknown dev option %s", key.c_str());
        }
    }
}

static bool load_driver() {
    if (driver.handle != nullptr) {
        return true;
    }
    if (api == nullptr || api->real_cuda_driver_path == nullptr) {
        PLUGIN_LOG("did not receive a real CUDA driver path.");
        return false;
    }
    driver.handle = dlopen(api->real_cuda_driver_path, RTLD_NOW | RTLD_LOCAL);
    if (driver.handle == nullptr) {
        PLUGIN_LOG("failed to load CUDA driver %s: %s",
                       api->real_cuda_driver_path, dlerror());
        return false;
    }

    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuCtxSynchronize, "cuCtxSynchronize", true, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuCtxGetCurrent, "cuCtxGetCurrent", true, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuCtxGetDevice, "cuCtxGetDevice", true, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuDeviceGetAttribute, "cuDeviceGetAttribute", true, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuStreamSynchronize, "cuStreamSynchronize", true, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuEventCreate, "cuEventCreate", true, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuEventRecord, "cuEventRecord", true, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuEventSynchronize, "cuEventSynchronize", true, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuEventElapsedTime, "cuEventElapsedTime", true, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuEventDestroy, "cuEventDestroy", false, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuEventDestroy_v2, "cuEventDestroy_v2", false, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuMemGetAllocationGranularity, "cuMemGetAllocationGranularity", true, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuMemCreate, "cuMemCreate", true, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuMemAddressReserve, "cuMemAddressReserve", true, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuMemMap, "cuMemMap", true, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuMemSetAccess, "cuMemSetAccess", true, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuMemExportToShareableHandle, "cuMemExportToShareableHandle", true, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuMemImportFromShareableHandle, "cuMemImportFromShareableHandle", true, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuMemUnmap, "cuMemUnmap", true, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuMemAddressFree, "cuMemAddressFree", true, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuMemRelease, "cuMemRelease", true, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuMemAlloc_v2, "cuMemAlloc_v2", true, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuMemFree_v2, "cuMemFree_v2", true, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuMemsetD8, "cuMemsetD8", false, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuMemsetD8_v2, "cuMemsetD8_v2", false, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuMemcpyHtoD, "cuMemcpyHtoD", false, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuMemcpyHtoD_v2, "cuMemcpyHtoD_v2", false, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuModuleGetGlobal, "cuModuleGetGlobal", false, false);
    PLUGIN_LOAD_CUDA_SYMBOL(&driver, pfn_cuModuleGetGlobal_v2, "cuModuleGetGlobal_v2", false, false);
    if (driver.pfn_cuEventDestroy == nullptr && driver.pfn_cuEventDestroy_v2 == nullptr) {
        PLUGIN_LOG("could not resolve cuEventDestroy.");
        return false;
    }
    if (driver.pfn_cuMemsetD8 == nullptr && driver.pfn_cuMemsetD8_v2 == nullptr) {
        PLUGIN_LOG("could not resolve cuMemsetD8.");
        return false;
    }
    if (driver.pfn_cuMemcpyHtoD == nullptr && driver.pfn_cuMemcpyHtoD_v2 == nullptr) {
        PLUGIN_LOG("could not resolve cuMemcpyHtoD.");
        return false;
    }
    if (driver.pfn_cuModuleGetGlobal == nullptr && driver.pfn_cuModuleGetGlobal_v2 == nullptr) {
        PLUGIN_LOG("could not resolve cuModuleGetGlobal.");
        return false;
    }
    return true;
}

static CUresult destroy_event(CUevent event) {
    if (event == nullptr) {
        return CUDA_SUCCESS;
    }
    if (driver.pfn_cuEventDestroy_v2 != nullptr) {
        return driver.pfn_cuEventDestroy_v2(event);
    }
    return driver.pfn_cuEventDestroy(event);
}

static bool cupti_ok(CUptiResult result, const char* call) {
    if (result == CUPTI_SUCCESS) {
        return true;
    }
    PLUGIN_LOG("%s failed: %s", call, cupti_status(result).c_str());
    return false;
}

static bool driver_ok(CUresult result, const char* call) {
    if (result == CUDA_SUCCESS) {
        return true;
    }
    PLUGIN_LOG("CUDA %s failed: %d", call, (int) result);
    return false;
}

static CUresult driver_memset_d8(CUdeviceptr ptr, unsigned char value, size_t bytes) {
    // Prefer the v2 driver ABI when exported.
    if (driver.pfn_cuMemsetD8_v2 != nullptr) {
        return driver.pfn_cuMemsetD8_v2(ptr, value, bytes);
    }
    return driver.pfn_cuMemsetD8(ptr, value, bytes);
}

static CUresult driver_memcpy_htod(CUdeviceptr dst, const void* src, size_t bytes) {
    // Prefer the v2 driver ABI when exported.
    if (driver.pfn_cuMemcpyHtoD_v2 != nullptr) {
        return driver.pfn_cuMemcpyHtoD_v2(dst, src, bytes);
    }
    return driver.pfn_cuMemcpyHtoD(dst, src, bytes);
}

static CUresult driver_module_get_global(CUmodule module, const char* name,
                                         CUdeviceptr* dptr, size_t* bytes) {
    // Resolve generated module globals through the real CUDA driver.
    if (driver.pfn_cuModuleGetGlobal_v2 != nullptr) {
        return driver.pfn_cuModuleGetGlobal_v2(dptr, bytes, module, name);
    }
    return driver.pfn_cuModuleGetGlobal(dptr, bytes, module, name);
}

static bool symbol_write(CUmodule module, const char* name,
                        const void* data, size_t bytes) {
    // Write one generated module symbol if present; example: globalResultBuff receives a scratch pointer.
    if (module == 0 || name == NULL || data == NULL || bytes == 0) {
        return false;
    }
    CUdeviceptr dst_ptr = 0;
    size_t dst_bytes = 0;
    if (driver_module_get_global(module, name, &dst_ptr, &dst_bytes) != CUDA_SUCCESS) {
        return false;
    }
    size_t copy_bytes = dst_bytes < bytes ? dst_bytes : bytes;
    return driver_memcpy_htod(dst_ptr, data, copy_bytes) == CUDA_SUCCESS;
}

static int prepare_mpi_locked() {
    // Lazily discover MPI rank/size; example: non-FabricPerf probes avoid this setup.
    if (fabric.mpi_prepared) {
        return 0;
    }
    int initialized = 0;
    PLUGIN_CHECK_MPI(MPI_Initialized(&initialized), "MPI_Initialized", -1);
    if (!initialized) {
        PLUGIN_LOG("requires MPI_Init before probe launches.");
        return -1;
    }
    PLUGIN_CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &fabric.world_rank), "MPI_Comm_rank", -1);
    PLUGIN_CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &fabric.world_size), "MPI_Comm_size", -1);
    if (fabric.world_size < 1 || fabric.world_size > FABRICPERF_MAX_CHANNEL) {
        PLUGIN_LOG("supports 1..%d MPI ranks, got %d.",
                       FABRICPERF_MAX_CHANNEL, fabric.world_size);
        return -1;
    }
    fabric.mpi_prepared = true;
    return 0;
}

static int create_fabric_buffer(size_t requested_size, CUdevice cu_dev,
                                FabricPerfSlot** ptr, size_t* mapped_size,
                                CUmemGenericAllocationHandle* handle,
                                CUmemFabricHandle* fabric_handle) {
    // Allocate one exportable CUDA VMM buffer; example: each rank exports its local follower buffer.
    CUmemAllocationProp prop;
    std::memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = cu_dev;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

    size_t granularity = 0;
    if (!driver_ok(driver.pfn_cuMemGetAllocationGranularity(
                       &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                   "cuMemGetAllocationGranularity")) {
        return -1;
    }
    *mapped_size = PLUGIN_ALIGN_UP(requested_size, granularity);

    if (!driver_ok(driver.pfn_cuMemCreate(handle, *mapped_size, &prop, 0), "cuMemCreate")) {
        return -1;
    }

    CUdeviceptr addr = 0;
    if (!driver_ok(driver.pfn_cuMemAddressReserve(&addr, *mapped_size, 0, 0, 0),
                   "cuMemAddressReserve") ||
        !driver_ok(driver.pfn_cuMemMap(addr, *mapped_size, 0, *handle, 0), "cuMemMap")) {
        return -1;
    }

    CUmemAccessDesc access;
    std::memset(&access, 0, sizeof(access));
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = cu_dev;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (!driver_ok(driver.pfn_cuMemSetAccess(addr, *mapped_size, &access, 1), "cuMemSetAccess") ||
        !driver_ok(driver.pfn_cuMemExportToShareableHandle(
                       fabric_handle, *handle, CU_MEM_HANDLE_TYPE_FABRIC, 0),
                   "cuMemExportToShareableHandle")) {
        return -1;
    }

    *ptr = reinterpret_cast<FabricPerfSlot*>(static_cast<uintptr_t>(addr));
    return driver_ok(driver_memset_d8(addr, 0, *mapped_size), "cuMemsetD8") ? 0 : -1;
}

static int import_fabric_buffer(const CUmemFabricHandle* fabric_handle,
                                size_t mapped_size, CUdevice cu_dev,
                                FabricPerfSlot** ptr,
                                CUmemGenericAllocationHandle* handle) {
    // Import one peer rank's fabric allocation; example: rank 0 imports rank 1's follower buffer.
    if (!driver_ok(driver.pfn_cuMemImportFromShareableHandle(
                       handle, const_cast<CUmemFabricHandle*>(fabric_handle), CU_MEM_HANDLE_TYPE_FABRIC),
                   "cuMemImportFromShareableHandle")) {
        return -1;
    }

    CUdeviceptr addr = 0;
    if (!driver_ok(driver.pfn_cuMemAddressReserve(&addr, mapped_size, 0, 0, 0),
                   "cuMemAddressReserve") ||
        !driver_ok(driver.pfn_cuMemMap(addr, mapped_size, 0, *handle, 0), "cuMemMap")) {
        return -1;
    }

    CUmemAccessDesc access;
    std::memset(&access, 0, sizeof(access));
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = cu_dev;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (!driver_ok(driver.pfn_cuMemSetAccess(addr, mapped_size, &access, 1), "cuMemSetAccess")) {
        return -1;
    }

    *ptr = reinterpret_cast<FabricPerfSlot*>(static_cast<uintptr_t>(addr));
    return 0;
}

static void unmap_fabric_buffer(FabricPerfSlot* ptr, size_t mapped_size) {
    // Release one virtual mapping while leaving handle release to the caller; example: peer mappings are unmapped before cuMemRelease.
    if (ptr == nullptr) {
        return;
    }
    CUdeviceptr addr = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr));
    driver.pfn_cuMemUnmap(addr, mapped_size);
    driver.pfn_cuMemAddressFree(addr, mapped_size);
}

static int prepare_buffers_locked() {
    // Lazily allocate FabricPerf cross-rank buffers; example: globalLeaderBuff first use triggers this path.
    if (fabric.buffers_prepared) {
        return 0;
    }
    if (prepare_mpi_locked() != 0) {
        return -1;
    }
    if (!driver_ok(driver.pfn_cuCtxGetDevice(&fabric.device), "cuCtxGetDevice")) {
        return -1;
    }

    int vmm_supported = 0;
    int fabric_supported = 0;
    if (!driver_ok(driver.pfn_cuDeviceGetAttribute(
                       &vmm_supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                       fabric.device),
                   "cuDeviceGetAttribute(VMM)") ||
        !driver_ok(driver.pfn_cuDeviceGetAttribute(
                       &fabric_supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                       fabric.device),
                   "cuDeviceGetAttribute(FABRIC)")) {
        return -1;
    }
    if (!vmm_supported || !fabric_supported) {
        PLUGIN_LOG("requires CUDA VMM/fabric support rank=%d vmm=%d fabric=%d.",
                       fabric.world_rank, vmm_supported, fabric_supported);
        return -1;
    }

    fabric.leader_handles = static_cast<CUmemGenericAllocationHandle*>(
        std::calloc(static_cast<size_t>(fabric.world_size), sizeof(CUmemGenericAllocationHandle)));
    fabric.follower_handles = static_cast<CUmemGenericAllocationHandle*>(
        std::calloc(static_cast<size_t>(fabric.world_size), sizeof(CUmemGenericAllocationHandle)));
    fabric.leader_buffs = static_cast<FabricPerfSlot**>(
        std::calloc(static_cast<size_t>(fabric.world_size), sizeof(FabricPerfSlot*)));
    fabric.follower_buffs = static_cast<FabricPerfSlot**>(
        std::calloc(static_cast<size_t>(fabric.world_size), sizeof(FabricPerfSlot*)));
    if (fabric.leader_handles == nullptr || fabric.follower_handles == nullptr ||
        fabric.leader_buffs == nullptr || fabric.follower_buffs == nullptr) {
        PLUGIN_LOG("failed to allocate host bookkeeping arrays.");
        return -1;
    }

    const size_t ptp_sequence_slots =
        FABRICPERF_PTP_SAMPLES_PER_LEADER * static_cast<size_t>(fabric.world_size);
    const size_t sequence_slots = ptp_sequence_slots + FABRICPERF_SENDRECV_MESSAGE_SLOTS;
    const size_t leader_requested_size =
        sequence_slots * static_cast<size_t>(fabric.world_size) *
        FABRICPERF_LEADER_SPACES * sizeof(FabricPerfSlot);
    const size_t follower_requested_size =
        sequence_slots * FABRICPERF_FOLLOWER_SPACES * sizeof(FabricPerfSlot);

    CUmemFabricHandle local_leader_handle;
    CUmemFabricHandle local_follower_handle;
    if (create_fabric_buffer(leader_requested_size, fabric.device,
                             &fabric.local_leader_buff,
                             &fabric.leader_alloc_size,
                             &fabric.leader_handles[fabric.world_rank],
                             &local_leader_handle) != 0 ||
        create_fabric_buffer(follower_requested_size, fabric.device,
                             &fabric.local_follower_buff,
                             &fabric.follower_alloc_size,
                             &fabric.follower_handles[fabric.world_rank],
                             &local_follower_handle) != 0) {
        return -1;
    }

    CUmemFabricHandle* leader_handles = static_cast<CUmemFabricHandle*>(
        std::malloc(static_cast<size_t>(fabric.world_size) * sizeof(CUmemFabricHandle)));
    CUmemFabricHandle* follower_handles = static_cast<CUmemFabricHandle*>(
        std::malloc(static_cast<size_t>(fabric.world_size) * sizeof(CUmemFabricHandle)));
    if (leader_handles == nullptr || follower_handles == nullptr) {
        std::free(leader_handles);
        std::free(follower_handles);
        PLUGIN_LOG("failed to allocate MPI fabric handle buffers.");
        return -1;
    }
    if (sizeof(CUmemFabricHandle) > static_cast<size_t>(INT_MAX)) {
        std::free(leader_handles);
        std::free(follower_handles);
        PLUGIN_LOG("CUmemFabricHandle is too large for MPI_Allgather.");
        return -1;
    }
    int mpi_status = MPI_Allgather(&local_leader_handle, static_cast<int>(sizeof(CUmemFabricHandle)), MPI_BYTE,
                                   leader_handles, static_cast<int>(sizeof(CUmemFabricHandle)), MPI_BYTE,
                                   MPI_COMM_WORLD);
    if (mpi_status != MPI_SUCCESS) {
        PLUGIN_LOG_MPI_ERROR(mpi_status, "MPI_Allgather(leader)");
        std::free(leader_handles);
        std::free(follower_handles);
        return -1;
    }
    mpi_status = MPI_Allgather(&local_follower_handle, static_cast<int>(sizeof(CUmemFabricHandle)), MPI_BYTE,
                               follower_handles, static_cast<int>(sizeof(CUmemFabricHandle)), MPI_BYTE,
                               MPI_COMM_WORLD);
    if (mpi_status != MPI_SUCCESS) {
        PLUGIN_LOG_MPI_ERROR(mpi_status, "MPI_Allgather(follower)");
        std::free(leader_handles);
        std::free(follower_handles);
        return -1;
    }

    for (int idx = 0; idx < fabric.world_size; idx++) {
        if (idx == fabric.world_rank) {
            fabric.leader_buffs[idx] = fabric.local_leader_buff;
            fabric.follower_buffs[idx] = fabric.local_follower_buff;
        } else if (import_fabric_buffer(&leader_handles[idx],
                                        fabric.leader_alloc_size,
                                        fabric.device,
                                        &fabric.leader_buffs[idx],
                                        &fabric.leader_handles[idx]) != 0 ||
                   import_fabric_buffer(&follower_handles[idx],
                                        fabric.follower_alloc_size,
                                        fabric.device,
                                        &fabric.follower_buffs[idx],
                                        &fabric.follower_handles[idx]) != 0) {
            std::free(leader_handles);
            std::free(follower_handles);
            return -1;
        }
    }
    std::free(leader_handles);
    std::free(follower_handles);

    if (!driver_ok(driver.pfn_cuMemAlloc_v2(&fabric.device_leader_table,
                                            static_cast<size_t>(fabric.world_size) * sizeof(FabricPerfSlot*)),
                   "cuMemAlloc(leader_table)") ||
        !driver_ok(driver_memcpy_htod(fabric.device_leader_table,
                                      fabric.leader_buffs,
                                      static_cast<size_t>(fabric.world_size) * sizeof(FabricPerfSlot*)),
                   "cuMemcpyHtoD(leader_table)") ||
        !driver_ok(driver.pfn_cuMemAlloc_v2(&fabric.device_follower_table,
                                            static_cast<size_t>(fabric.world_size) * sizeof(FabricPerfSlot*)),
                   "cuMemAlloc(follower_table)") ||
        !driver_ok(driver_memcpy_htod(fabric.device_follower_table,
                                      fabric.follower_buffs,
                                      static_cast<size_t>(fabric.world_size) * sizeof(FabricPerfSlot*)),
                   "cuMemcpyHtoD(follower_table)")) {
        return -1;
    }

    fabric.result_buffer_size = FABRICPERF_RESULT_SLOTS * sizeof(FabricPerfSlot);
    if (!driver_ok(driver.pfn_cuMemAlloc_v2(&fabric.device_result_buffer,
                                            fabric.result_buffer_size),
                   "cuMemAlloc(result_buffer)") ||
        !driver_ok(driver_memset_d8(fabric.device_result_buffer, 0,
                                    fabric.result_buffer_size),
                   "cuMemsetD8(result_buffer)")) {
        return -1;
    }

    fabric.buffers_prepared = true;
    if (!fabric.reported) {
        PLUGIN_LOG("prepared rank=%d world=%d device=%d",
                       fabric.world_rank, fabric.world_size, fabric.device);
        fabric.reported = true;
    }
    return 0;
}

static void teardown_fabric_runtime() {
    // Release FabricPerf VMM resources before closing libcuda; example: plugin shutdown after trace analysis.
    std::lock_guard<std::mutex> lock(fabric_mutex);
    if (fabric.buffers_prepared && driver.pfn_cuCtxSynchronize != nullptr) {
        driver.pfn_cuCtxSynchronize();
    }
    if (fabric.device_leader_table != 0 && driver.pfn_cuMemFree_v2 != nullptr) {
        driver.pfn_cuMemFree_v2(fabric.device_leader_table);
    }
    if (fabric.device_follower_table != 0 && driver.pfn_cuMemFree_v2 != nullptr) {
        driver.pfn_cuMemFree_v2(fabric.device_follower_table);
    }
    if (fabric.device_result_buffer != 0 && driver.pfn_cuMemFree_v2 != nullptr) {
        driver.pfn_cuMemFree_v2(fabric.device_result_buffer);
    }
    if (fabric.leader_buffs != nullptr) {
        for (int idx = 0; idx < fabric.world_size; idx++) {
            unmap_fabric_buffer(fabric.leader_buffs[idx], fabric.leader_alloc_size);
        }
        std::free(fabric.leader_buffs);
    }
    if (fabric.follower_buffs != nullptr) {
        for (int idx = 0; idx < fabric.world_size; idx++) {
            unmap_fabric_buffer(fabric.follower_buffs[idx], fabric.follower_alloc_size);
        }
        std::free(fabric.follower_buffs);
    }
    if (fabric.leader_handles != nullptr) {
        for (int idx = 0; idx < fabric.world_size; idx++) {
            if (fabric.leader_handles[idx] != 0 && driver.pfn_cuMemRelease != nullptr) {
                driver.pfn_cuMemRelease(fabric.leader_handles[idx]);
            }
        }
        std::free(fabric.leader_handles);
    }
    if (fabric.follower_handles != nullptr) {
        for (int idx = 0; idx < fabric.world_size; idx++) {
            if (fabric.follower_handles[idx] != 0 && driver.pfn_cuMemRelease != nullptr) {
                driver.pfn_cuMemRelease(fabric.follower_handles[idx]);
            }
        }
        std::free(fabric.follower_handles);
    }
    fabric = {};
}

static int parse_rank() {
    {
        // Prefer FabricPerf's MPI rank when probe symbols already prepared it; example: one-pass MPI runs set rank before PM starts.
        std::lock_guard<std::mutex> lock(fabric_mutex);
        if (fabric.mpi_prepared) {
            return fabric.world_rank;
        }
    }
    const char* names[] = {
        "FABRICPERF_RANK",
        "OMPI_COMM_WORLD_RANK",
        "PMIX_RANK",
        "PMI_RANK",
        "SLURM_PROCID",
        "MV2_COMM_WORLD_RANK",
        nullptr,
    };
    for (int i = 0; names[i] != nullptr; ++i) {
        const char* raw = std::getenv(names[i]);
        if (raw != nullptr && raw[0] != '\0') {
            return std::atoi(raw);
        }
    }
    return 0;
}

static std::string trace_dir() {
    if (api != nullptr && api->api_size >= sizeof(neutrino_plugin_api_v1) &&
        api->trace_dir != nullptr && api->trace_dir[0] != '\0') {
        return api->trace_dir;
    }
    const char* active = std::getenv("NEUTRINO_ACTIVE_TRACEDIR");
    if (active != nullptr && active[0] != '\0') {
        return active;
    }
    const char* parent = std::getenv("NEUTRINO_TRACEDIR");
    return parent != nullptr && parent[0] != '\0' ? parent : ".";
}

static std::string csv_path_locked() {
    if (state.csv_path.empty()) {
        state.csv_path = trace_dir() + "/fabricperf_cupti.csv";
    }
    return state.csv_path;
}

static bool open_csv_locked() {
    if (state.csv != nullptr) {
        return true;
    }
    state.csv_path = csv_path_locked();
    struct stat st;
    bool write_header = stat(state.csv_path.c_str(), &st) != 0 || st.st_size == 0;
    state.csv = std::fopen(state.csv_path.c_str(), "a");
    if (state.csv == nullptr) {
        PLUGIN_LOG("could not open %s", state.csv_path.c_str());
        return false;
    }
    if (write_header) {
        std::fprintf(state.csv,
                     "rank,device,launch_index,kernel,grid,block,shared_bytes,duration_s,"
                     "dram_read_Bps,dram_write_Bps,nvlink_rx_Bps,nvlink_tx_Bps,"
                     "xbar_read_Bps,xbar_write_Bps,xbar_metric,xbar_value,raw_metrics\n");
        std::fflush(state.csv);
    }
    return true;
}

static bool create_host_object_locked(CUpti_Profiler_Host_Object** out) {
    CUpti_Profiler_Host_Initialize_Params params = {CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
    params.profilerType = CUPTI_PROFILER_TYPE_PM_SAMPLING;
    params.pChipName = state.chip_name.c_str();
    params.pCounterAvailabilityImage =
        state.counter_availability.empty() ? nullptr : state.counter_availability.data();
    CUptiResult result = cuptiProfilerHostInitialize(&params);
    if (result != CUPTI_SUCCESS) {
        PLUGIN_LOG("host initialize failed: %s", cupti_status(result).c_str());
        return false;
    }
    *out = params.pHostObject;
    return true;
}

static void destroy_host_object(CUpti_Profiler_Host_Object* host) {
    if (host == nullptr) {
        return;
    }
    CUpti_Profiler_Host_Deinitialize_Params params = {CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
    params.pHostObject = host;
    cuptiProfilerHostDeinitialize(&params);
}

static CUptiResult metric_config_status_locked(const std::string& metric) {
    CUpti_Profiler_Host_Object* host = nullptr;
    if (!create_host_object_locked(&host)) {
        return CUPTI_ERROR_UNKNOWN;
    }
    const char* name = metric.c_str();
    CUpti_Profiler_Host_ConfigAddMetrics_Params params =
        {CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
    params.pHostObject = host;
    params.ppMetricNames = &name;
    params.numMetrics = 1;
    CUptiResult result = cuptiProfilerHostConfigAddMetrics(&params);
    destroy_host_object(host);
    return result;
}

static bool build_config_image_locked(const std::vector<std::string>& metrics,
                                      CUpti_Profiler_Host_Object** host_out,
                                      std::vector<uint8_t>* config_out,
                                      size_t* passes_out) {
    *host_out = nullptr;
    config_out->clear();
    if (passes_out != nullptr) {
        *passes_out = 0;
    }
    CUpti_Profiler_Host_Object* host = nullptr;
    if (!create_host_object_locked(&host)) {
        return false;
    }
    std::vector<const char*> metric_ptrs;
    metric_ptrs.reserve(metrics.size());
    for (const std::string& metric : metrics) {
        metric_ptrs.push_back(metric.c_str());
    }

    CUpti_Profiler_Host_ConfigAddMetrics_Params add_params =
        {CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
    add_params.pHostObject = host;
    add_params.ppMetricNames = metric_ptrs.data();
    add_params.numMetrics = metric_ptrs.size();
    if (!cupti_ok(cuptiProfilerHostConfigAddMetrics(&add_params), "cuptiProfilerHostConfigAddMetrics")) {
        destroy_host_object(host);
        return false;
    }

    CUpti_Profiler_Host_GetConfigImageSize_Params size_params =
        {CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE};
    size_params.pHostObject = host;
    if (!cupti_ok(cuptiProfilerHostGetConfigImageSize(&size_params),
                  "cuptiProfilerHostGetConfigImageSize")) {
        destroy_host_object(host);
        return false;
    }
    config_out->assign(size_params.configImageSize, 0);

    CUpti_Profiler_Host_GetConfigImage_Params image_params =
        {CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE};
    image_params.pHostObject = host;
    image_params.pConfigImage = config_out->data();
    image_params.configImageSize = config_out->size();
    if (!cupti_ok(cuptiProfilerHostGetConfigImage(&image_params),
                  "cuptiProfilerHostGetConfigImage")) {
        destroy_host_object(host);
        return false;
    }

    CUpti_Profiler_Host_GetNumOfPasses_Params passes_params =
        {CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE};
    passes_params.pConfigImage = config_out->data();
    passes_params.configImageSize = config_out->size();
    if (!cupti_ok(cuptiProfilerHostGetNumOfPasses(&passes_params),
                  "cuptiProfilerHostGetNumOfPasses")) {
        destroy_host_object(host);
        return false;
    }
    if (passes_out != nullptr) {
        *passes_out = passes_params.numOfPasses;
    }
    *host_out = host;
    return true;
}

static bool create_config_image_locked() {
    if (state.supported_metrics.empty()) {
        PLUGIN_LOG("has no supported requested metrics; CSV rows will contain event duration and unsupported metric statuses.");
        return true;
    }

    size_t passes = 0;
    if (!build_config_image_locked(state.supported_metrics, &state.host_object,
                                   &state.config_image, &passes)) {
        return false;
    }
    if (passes > 1) {
        PLUGIN_LOG("PM sampling config needs %zu passes; keeping only single-pass metrics",
                       passes);
        destroy_host_object(state.host_object);
        state.host_object = nullptr;
        state.config_image.clear();

        std::vector<std::string> selected;
        for (const std::string& metric : state.supported_metrics) {
            std::vector<std::string> candidate = selected;
            candidate.push_back(metric);
            CUpti_Profiler_Host_Object* test_host = nullptr;
            std::vector<uint8_t> test_config;
            size_t test_passes = 0;
            bool ok = build_config_image_locked(candidate, &test_host, &test_config, &test_passes);
            if (test_host != nullptr) {
                destroy_host_object(test_host);
            }
            if (ok && test_passes == 1) {
                selected.push_back(metric);
            } else {
                state.unsupported_metrics[metric] = CUPTI_ERROR_NOT_SUPPORTED;
                PLUGIN_LOG("PM sampling skipped multi-pass metric %s",
                               metric.c_str());
            }
        }
        state.supported_metrics = selected;
        if (state.supported_metrics.empty()) {
            PLUGIN_LOG("PM sampling has no single-pass requested metrics; writing duration-only rows.");
            return true;
        }
        if (!build_config_image_locked(state.supported_metrics, &state.host_object,
                                       &state.config_image, &passes)) {
            return false;
        }
    }

    if (state.host_object != nullptr) {
        PLUGIN_LOG("PM sampling config metrics=%zu passes=%zu",
                       state.supported_metrics.size(), passes);
    }
    return true;
}

static void mark_requested_metrics_unavailable_locked(CUptiResult status) {
    if (state.requested_metrics.empty()) {
        // Preserve CSV columns even when CUPTI cannot provide counter availability.
        state.requested_metrics = default_pm_metrics();
    }
    state.supported_metrics.clear();
    state.values.clear();
    state.eval_errors.clear();
    for (const std::string& metric : state.requested_metrics) {
        state.unsupported_metrics[metric] = status;
    }
}

static bool initialize_profiler_once_locked() {
    if (state.profiler_initialized) {
        return true;
    }
    CUpti_Profiler_Initialize_Params params = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    if (!cupti_ok(cuptiProfilerInitialize(&params), "cuptiProfilerInitialize")) {
        return false;
    }
    state.profiler_initialized = true;
    return true;
}

static void teardown_context_locked() {
    if (state.host_object != nullptr) {
        destroy_host_object(state.host_object);
        state.host_object = nullptr;
    }
    if (state.pm_sampling != nullptr) {
        if (state.pm_disable_on_teardown) {
            CUpti_PmSampling_Disable_Params disable_params =
                {CUpti_PmSampling_Disable_Params_STRUCT_SIZE};
            disable_params.pPmSamplingObject = state.pm_sampling;
            cuptiPmSamplingDisable(&disable_params);
        } else {
            PLUGIN_LOG("PM sampling disable skipped to avoid CUDA 12.8 teardown double-free");
        }
        state.pm_sampling = nullptr;
    }
    if (state.start_event != nullptr) {
        destroy_event(state.start_event);
        state.start_event = nullptr;
    }
    if (state.end_event != nullptr) {
        destroy_event(state.end_event);
        state.end_event = nullptr;
    }
    state.ready = false;
    state.context = nullptr;
    state.device = -1;
    state.chip_name.clear();
    state.counter_availability.clear();
    state.config_image.clear();
    state.counter_data.clear();
    state.requested_metrics.clear();
    state.supported_metrics.clear();
    state.unsupported_metrics.clear();
    state.active_pm_sampling = false;
    state.pm_overflow = false;
    state.pm_total_samples = 0;
    state.pm_populated_samples = 0;
    state.pm_completed_samples = 0;
}

static bool setup_context_locked(CUcontext ctx) {
    if (state.ready && state.context == ctx) {
        return true;
    }
    teardown_context_locked();
    state.context = ctx;
    state.rank = parse_rank();

    if (!initialize_profiler_once_locked()) {
        return false;
    }

    if (!driver_ok(driver.pfn_cuCtxGetDevice(&state.device), "cuCtxGetDevice")) {
        return false;
    }

    CUpti_Device_GetChipName_Params chip_params = {CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    chip_params.deviceIndex = (size_t) state.device;
    if (!cupti_ok(cuptiDeviceGetChipName(&chip_params), "cuptiDeviceGetChipName")) {
        return false;
    }
    state.chip_name = chip_params.pChipName != nullptr ? chip_params.pChipName : "";

    CUpti_PmSampling_GetCounterAvailability_Params avail_params =
        {CUpti_PmSampling_GetCounterAvailability_Params_STRUCT_SIZE};
    avail_params.deviceIndex = (size_t) state.device;
    CUptiResult availability_status = cuptiPmSamplingGetCounterAvailability(&avail_params);
    if (availability_status != CUPTI_SUCCESS) {
        PLUGIN_LOG("PM counter availability unavailable: %s; writing duration-only rows",
                       cupti_status(availability_status).c_str());
        mark_requested_metrics_unavailable_locked(availability_status);
        goto create_events;
    }
    state.counter_availability.assign(avail_params.counterAvailabilityImageSize, 0);
    avail_params.pCounterAvailabilityImage = state.counter_availability.data();
    availability_status = cuptiPmSamplingGetCounterAvailability(&avail_params);
    if (availability_status != CUPTI_SUCCESS) {
        PLUGIN_LOG("PM counter availability unavailable: %s; writing duration-only rows",
                       cupti_status(availability_status).c_str());
        mark_requested_metrics_unavailable_locked(availability_status);
        goto create_events;
    }

    // Reset requested PM metrics for each new CUDA context.
    state.requested_metrics = default_pm_metrics();
    for (const std::string& metric : state.requested_metrics) {
        CUptiResult status = metric_config_status_locked(metric);
        if (status == CUPTI_SUCCESS) {
            add_unique(state.supported_metrics, metric);
        } else {
            state.unsupported_metrics[metric] = status;
            PLUGIN_LOG("unsupported metric %s: %s",
                           metric.c_str(), cupti_status(status).c_str());
        }
    }

    if (!create_config_image_locked()) {
        return false;
    }

    if (!state.supported_metrics.empty()) {
        CUpti_PmSampling_Enable_Params enable_params =
            {CUpti_PmSampling_Enable_Params_STRUCT_SIZE};
        enable_params.deviceIndex = (size_t) state.device;
        if (!cupti_ok(cuptiPmSamplingEnable(&enable_params), "cuptiPmSamplingEnable")) {
            return false;
        }
        state.pm_sampling = enable_params.pPmSamplingObject;
    }

create_events:
    if (!driver_ok(driver.pfn_cuEventCreate(&state.start_event, CU_EVENT_DEFAULT), "cuEventCreate(start)") ||
        !driver_ok(driver.pfn_cuEventCreate(&state.end_event, CU_EVENT_DEFAULT), "cuEventCreate(end)")) {
        return false;
    }

    state.ready = true;
    PLUGIN_LOG("ready backend=%s rank=%d device=%d chip=%s csv=%s",
                   CUPTI_BACKEND_NAME,
                   state.rank, state.device, state.chip_name.c_str(),
                   csv_path_locked().c_str());
    return true;
}

static bool prepare_counter_data_locked() {
    state.counter_data.clear();
    if (state.supported_metrics.empty()) {
        return true;
    }

    std::vector<const char*> metric_ptrs;
    metric_ptrs.reserve(state.supported_metrics.size());
    for (const std::string& metric : state.supported_metrics) {
        metric_ptrs.push_back(metric.c_str());
    }

    CUpti_PmSampling_GetCounterDataSize_Params size_params =
        {CUpti_PmSampling_GetCounterDataSize_Params_STRUCT_SIZE};
    size_params.pPmSamplingObject = state.pm_sampling;
    size_params.pMetricNames = metric_ptrs.data();
    size_params.numMetrics = metric_ptrs.size();
    size_params.maxSamples = state.pm_max_samples;
    if (!cupti_ok(cuptiPmSamplingGetCounterDataSize(&size_params),
                  "cuptiPmSamplingGetCounterDataSize")) {
        return false;
    }
    state.counter_data.assign(size_params.counterDataSize, 0);

    CUpti_PmSampling_CounterDataImage_Initialize_Params init_params =
        {CUpti_PmSampling_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    init_params.pPmSamplingObject = state.pm_sampling;
    init_params.counterDataSize = state.counter_data.size();
    init_params.pCounterData = state.counter_data.data();
    if (!cupti_ok(cuptiPmSamplingCounterDataImageInitialize(&init_params),
                  "cuptiPmSamplingCounterDataImageInitialize")) {
        return false;
    }

    CUpti_PmSampling_SetConfig_Params set_params =
        {CUpti_PmSampling_SetConfig_Params_STRUCT_SIZE};
    set_params.pPmSamplingObject = state.pm_sampling;
    set_params.configSize = state.config_image.size();
    set_params.pConfig = state.config_image.data();
    set_params.hardwareBufferSize = state.pm_hardware_buffer_size;
    set_params.samplingInterval = state.pm_sampling_interval;
    set_params.triggerMode = state.pm_trigger_mode;
    set_params.hwBufferAppendMode = state.pm_append_mode;
    return cupti_ok(cuptiPmSamplingSetConfig(&set_params), "cuptiPmSamplingSetConfig");
}

static void evaluate_pm_metrics_locked() {
    state.values.clear();
    state.eval_errors.clear();
    state.pm_overflow = false;
    state.pm_total_samples = 0;
    state.pm_populated_samples = 0;
    state.pm_completed_samples = 0;
    if (state.supported_metrics.empty() || state.counter_data.empty()) {
        return;
    }

    CUpti_PmSampling_DecodeData_Params decode_params =
        {CUpti_PmSampling_DecodeData_Params_STRUCT_SIZE};
    decode_params.pPmSamplingObject = state.pm_sampling;
    decode_params.pCounterDataImage = state.counter_data.data();
    decode_params.counterDataImageSize = state.counter_data.size();
    CUptiResult decode_status = cuptiPmSamplingDecodeData(&decode_params);
    state.pm_overflow = decode_params.overflow != 0;
    if (decode_status != CUPTI_SUCCESS) {
        for (const std::string& metric : state.supported_metrics) {
            state.eval_errors[metric] = decode_status;
        }
        PLUGIN_LOG("PM decode failed: %s", cupti_status(decode_status).c_str());
        return;
    }

    CUpti_PmSampling_GetCounterDataInfo_Params info_params =
        {CUpti_PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE};
    info_params.pCounterDataImage = state.counter_data.data();
    info_params.counterDataImageSize = state.counter_data.size();
    CUptiResult info_status = cuptiPmSamplingGetCounterDataInfo(&info_params);
    if (info_status != CUPTI_SUCCESS) {
        for (const std::string& metric : state.supported_metrics) {
            state.eval_errors[metric] = info_status;
        }
        PLUGIN_LOG("PM counter data info failed: %s",
                       cupti_status(info_status).c_str());
        return;
    }
    state.pm_total_samples = info_params.numTotalSamples;
    state.pm_populated_samples = info_params.numPopulatedSamples;
    state.pm_completed_samples = info_params.numCompletedSamples;
    if (state.pm_completed_samples == 0) {
        for (const std::string& metric : state.supported_metrics) {
            state.eval_errors[metric] = CUPTI_ERROR_UNKNOWN;
        }
        PLUGIN_LOG("PM counter data has no completed samples");
        return;
    }

    std::unordered_map<std::string, size_t> value_counts;
    for (size_t sample_index = 0; sample_index < state.pm_completed_samples; ++sample_index) {
        CUpti_PmSampling_CounterData_GetSampleInfo_Params sample_params =
            {CUpti_PmSampling_CounterData_GetSampleInfo_Params_STRUCT_SIZE};
        sample_params.pPmSamplingObject = state.pm_sampling;
        sample_params.pCounterDataImage = state.counter_data.data();
        sample_params.counterDataImageSize = state.counter_data.size();
        sample_params.sampleIndex = sample_index;
        CUptiResult sample_status = cuptiPmSamplingCounterDataGetSampleInfo(&sample_params);
        if (sample_status != CUPTI_SUCCESS) {
            PLUGIN_LOG("PM sample info failed for sample %zu: %s",
                           sample_index, cupti_status(sample_status).c_str());
        }

        for (const std::string& metric : state.supported_metrics) {
            const char* name = metric.c_str();
            double value = 0.0;
            CUpti_Profiler_Host_EvaluateToGpuValues_Params eval_params =
                {CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
            eval_params.pHostObject = state.host_object;
            eval_params.pCounterDataImage = state.counter_data.data();
            eval_params.counterDataImageSize = state.counter_data.size();
            eval_params.ppMetricNames = &name;
            eval_params.numMetrics = 1;
            eval_params.rangeIndex = sample_index;
            eval_params.pMetricValues = &value;
            CUptiResult eval_status = cuptiProfilerHostEvaluateToGpuValues(&eval_params);
            if (eval_status == CUPTI_SUCCESS) {
                state.values[metric] += value;
                value_counts[metric] += 1;
            } else if (value_counts.find(metric) == value_counts.end()) {
                state.eval_errors[metric] = eval_status;
            }
        }
    }

    for (const auto& kv : value_counts) {
        const std::string& metric = kv.first;
        // Average ratio/throughput-style metrics across completed PM samples.
        const bool average_metric =
            metric.find(".avg") != std::string::npos ||
            metric.find("pct_of_peak") != std::string::npos ||
            metric.find("throughput") != std::string::npos;
        if (kv.second != 0 && average_metric) {
            state.values[metric] /= (double) kv.second;
        }
        state.eval_errors.erase(metric);
    }
}

static std::string double_field(double value) {
    if (!(value == value)) {
        return "";
    }
    std::ostringstream os;
    os << std::setprecision(17) << value;
    return os.str();
}

static std::string csv_escape(const std::string& value) {
    bool quote = value.find_first_of(",\"\n\r") != std::string::npos;
    if (!quote) {
        return value;
    }
    std::string out = "\"";
    for (char c : value) {
        if (c == '"') {
            out += "\"\"";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('"');
    return out;
}

static bool metric_value(const std::string& name, double* value) {
    auto it = state.values.find(name);
    if (it == state.values.end()) {
        return false;
    }
    *value = it->second;
    return true;
}

static double duration_seconds_locked() {
    double metric_duration = 0.0;
    if (metric_value("gpu__time_duration.sum", &metric_duration) && metric_duration > 0.0) {
        return metric_duration * 1.0e-9;
    }
    return state.event_duration_s;
}

static std::string bps_field(const std::string& metric, double duration_s) {
    double bytes = 0.0;
    if (duration_s <= 0.0 || !metric_value(metric, &bytes)) {
        return "";
    }
    return double_field(bytes / duration_s);
}

static std::string raw_metrics_field_locked() {
    std::ostringstream os;
    bool first = true;
    for (const std::string& metric : state.requested_metrics) {
        if (!first) {
            os << ";";
        }
        first = false;
        os << metric << "=";
        auto value_it = state.values.find(metric);
        if (value_it != state.values.end()) {
            os << std::setprecision(17) << value_it->second;
            continue;
        }
        auto unsupported_it = state.unsupported_metrics.find(metric);
        if (unsupported_it != state.unsupported_metrics.end()) {
            os << "UNSUPPORTED:" << cupti_status(unsupported_it->second);
            continue;
        }
        auto eval_it = state.eval_errors.find(metric);
        if (eval_it != state.eval_errors.end()) {
            os << "EVAL_ERROR:" << cupti_status(eval_it->second);
            continue;
        }
        os << "MISSING";
    }
    os << ";cupti_backend=" << CUPTI_BACKEND_NAME;
    os << ";pm_samples_total=" << state.pm_total_samples;
    os << ";pm_samples_populated=" << state.pm_populated_samples;
    os << ";pm_samples_completed=" << state.pm_completed_samples;
    os << ";pm_overflow=" << (state.pm_overflow ? 1 : 0);
    os << ";pm_interval=" << state.pm_sampling_interval;
    os << ";event_duration_s=" << std::setprecision(17) << state.event_duration_s;
    return os.str();
}

static void write_csv_row_locked(CUresult launch_result) {
    if (state.csv == nullptr) {
        return;
    }

    double duration_s = duration_seconds_locked();
    std::ostringstream grid;
    grid << state.launch.grid_dim[0] << "," << state.launch.grid_dim[1] << "," << state.launch.grid_dim[2];
    std::ostringstream block;
    block << state.launch.block_dim[0] << "," << state.launch.block_dim[1] << "," << state.launch.block_dim[2];

    std::vector<std::string> fields = {
        std::to_string(state.rank),
        std::to_string((int) state.device),
        std::to_string(state.launch.launch_index),
        state.launch.kernel_name != nullptr ? state.launch.kernel_name : "",
        grid.str(),
        block.str(),
        std::to_string(state.launch.shared_mem_bytes),
        duration_s > 0.0 ? double_field(duration_s) : "",
        bps_field("dram__bytes_read.sum", duration_s),
        bps_field("dram__bytes_write.sum", duration_s),
        bps_field("nvlrx__bytes.sum", duration_s),
        bps_field("nvltx__bytes.sum", duration_s),
        "",
        "",
        "",
        "",
        raw_metrics_field_locked(),
    };

    if (launch_result != CUDA_SUCCESS) {
        fields.back() += ";launch_result=CUDA_" + std::to_string((int) launch_result);
    }

    for (size_t i = 0; i < fields.size(); ++i) {
        if (i != 0) {
            std::fputc(',', state.csv);
        }
        std::string escaped = csv_escape(fields[i]);
        std::fputs(escaped.c_str(), state.csv);
    }
    std::fputc('\n', state.csv);
    std::fflush(state.csv);
}

extern "C" int neutrino_plugin_init_v1(const neutrino_plugin_api_v1* plugin_api) {
    api = plugin_api;
    PLUGIN_REQUIRE_API(-1);

    // Apply cold-path developer options once after ABI validation.
    configure_dev_options_locked();
    if (!load_driver()) {
        return -1;
    }
    PLUGIN_LOG("initialized backend=%s", CUPTI_BACKEND_NAME);
    return 0;
}

extern "C" void neutrino_plugin_fini_v1(void) {
    teardown_fabric_runtime();
    std::lock_guard<std::mutex> lock(state.mutex);
    teardown_context_locked();
    if (state.csv != nullptr) {
        std::fclose(state.csv);
        state.csv = nullptr;
    }
    if (driver.handle != nullptr) {
        dlclose(driver.handle);
        driver = {};
    }
}

extern "C" int neutrino_plugin_prepare_launch_v1(
    const char* kernel_name,
    const neutrino_plugin_symbol_v1* symbols,
    int n_symbols,
    unsigned int launch_index,
    const neutrino_plugin_module_context_v1* context) {
    // Prepare FabricPerf symbols before Neutrino writes generated module globals; example: globalLeaderBuff triggers VMM setup.
    (void) kernel_name;
    (void) launch_index;
    (void) context;
    bool needs_mpi = false;
    bool needs_buffers = false;
    for (int idx = 0; idx < n_symbols; idx++) {
        const char* name = symbols[idx].name;
        // Buffer-backed FabricPerf symbols require MPI rank discovery plus peer VMM setup.
        const bool buffer_symbol =
            PLUGIN_IS_SYMBOL(name, "globalLeaderBuff") != 0 ||
            PLUGIN_IS_SYMBOL(name, "globalFollowerBuff") != 0 ||
            PLUGIN_IS_SYMBOL(name, "globalResultBuff") != 0 ||
            PLUGIN_IS_SYMBOL(name, "latencyGlobal") != 0;
        needs_buffers = needs_buffers || buffer_symbol;

        // Rank-only symbols need MPI discovery but not CUDA fabric buffer exchange.
        needs_mpi = needs_mpi ||
                    buffer_symbol ||
                    PLUGIN_IS_SYMBOL(name, "deviceId") != 0 ||
                    PLUGIN_IS_SYMBOL(name, "numDevices") != 0;
    }
    if (needs_buffers) {
        // Serialize lazy FabricPerf buffer setup; example: globalLeaderBuff needs peer VMM mappings.
        std::lock_guard<std::mutex> lock(fabric_mutex);
        return prepare_buffers_locked();
    }
    if (needs_mpi) {
        // Serialize lazy MPI rank setup; example: deviceId maps to fabric.world_rank.
        std::lock_guard<std::mutex> lock(fabric_mutex);
        return prepare_mpi_locked();
    }
    return 0;
}

extern "C" int neutrino_plugin_resolve_symbol_v1(
    const neutrino_plugin_symbol_v1* symbol,
    void* destination_module,
    const neutrino_plugin_module_context_v1* context) {
    // Resolve FabricPerf-owned symbols and delegate generic symbols to Neutrino; example: unknown user symbols use core fallback.
    if (symbol == nullptr || symbol->name == nullptr) {
        return 0;
    }
    CUmodule destination = (CUmodule) destination_module;
    const char* name = symbol->name;
    if (PLUGIN_IS_SYMBOL(name, "deviceId") != 0) {
        {
            // Prepare rank metadata before reading fabric.world_rank.
            std::lock_guard<std::mutex> lock(fabric_mutex);
            if (prepare_mpi_locked() != 0) return -1;
        }
        uint32_t value = static_cast<uint32_t>(fabric.world_rank);
        return symbol_write(destination, name, &value, sizeof(value)) ? 1 : -1;
    }
    if (PLUGIN_IS_SYMBOL(name, "numDevices") != 0) {
        {
            // Prepare rank metadata before reading fabric.world_size.
            std::lock_guard<std::mutex> lock(fabric_mutex);
            if (prepare_mpi_locked() != 0) return -1;
        }
        uint32_t value = static_cast<uint32_t>(fabric.world_size);
        return symbol_write(destination, name, &value, sizeof(value)) ? 1 : -1;
    }
    if (PLUGIN_IS_SYMBOL(name, "hostId") != 0) {
        const char* raw = std::getenv("FABRICPERF_HOSTID");
        uint32_t value = raw != nullptr ? static_cast<uint32_t>(std::atoi(raw)) : 0u;
        return symbol_write(destination, name, &value, sizeof(value)) ? 1 : -1;
    }
    if (PLUGIN_IS_SYMBOL(name, "ptpRunId") != 0) {
        uint32_t value = context != nullptr ? context->launch_index : 0u;
        return symbol_write(destination, name, &value, sizeof(value)) ? 1 : -1;
    }
    if (PLUGIN_IS_SYMBOL(name, "launchIndex") != 0) {
        // Publish the memory probe merge key; example: analyzer joins PM rows to probe bytes by launch_index.
        uint32_t value = context != nullptr ? context->launch_index : 0u;
        return symbol_write(destination, name, &value, sizeof(value)) ? 1 : -1;
    }
    if (PLUGIN_IS_SYMBOL(name, "ptpGlobalBarrier") != 0 ||
        PLUGIN_IS_SYMBOL(name, "ptpGlobalBarrierSense") != 0) {
        uint32_t value = 0u;
        return symbol_write(destination, name, &value, sizeof(value)) ? 1 : -1;
    }
    if (PLUGIN_IS_SYMBOL(name, "globalFollowerBuff") != 0) {
        {
            // Prepare peer pointer tables before publishing the follower table symbol.
            std::lock_guard<std::mutex> lock(fabric_mutex);
            if (prepare_buffers_locked() != 0) return -1;
        }
        return symbol_write(destination, name,
                            &fabric.device_follower_table,
                            sizeof(fabric.device_follower_table)) ? 1 : -1;
    }
    if (PLUGIN_IS_SYMBOL(name, "globalLeaderBuff") != 0) {
        {
            // Prepare peer pointer tables before publishing the leader table symbol.
            std::lock_guard<std::mutex> lock(fabric_mutex);
            if (prepare_buffers_locked() != 0) return -1;
        }
        return symbol_write(destination, name,
                            &fabric.device_leader_table,
                            sizeof(fabric.device_leader_table)) ? 1 : -1;
    }
    if (PLUGIN_IS_SYMBOL(name, "globalResultBuff") != 0 ||
        PLUGIN_IS_SYMBOL(name, "latencyGlobal") != 0) {
        {
            // Prepare result scratch storage before clearing and publishing it.
            std::lock_guard<std::mutex> lock(fabric_mutex);
            if (prepare_buffers_locked() != 0) return -1;
        }
        if (driver_memset_d8(fabric.device_result_buffer, 0,
                             fabric.result_buffer_size) != CUDA_SUCCESS) {
            return -1;
        }
        return symbol_write(destination, name,
                            &fabric.device_result_buffer,
                            sizeof(fabric.device_result_buffer)) ? 1 : -1;
    }
    return 0;
}

extern "C" int neutrino_plugin_begin_launch_v1(
    const neutrino_plugin_launch_context_v1* context) {
    state.mutex.lock();
    state.active = false;
    state.active_pm_sampling = false;
    state.active_events = false;
    state.event_duration_s = 0.0;
    state.pm_overflow = false;
    state.pm_total_samples = 0;
    state.pm_populated_samples = 0;
    state.pm_completed_samples = 0;
    state.values.clear();
    state.eval_errors.clear();

    if (context != nullptr) {
        state.launch = *context;
    } else {
        state.launch = {};
    }

    CUcontext ctx = nullptr;
    // Read the active CUDA context before creating CUPTI objects for this launch.
    if (!driver_ok(driver.pfn_cuCtxGetCurrent(&ctx), "cuCtxGetCurrent") ||
        ctx == nullptr ||
        !open_csv_locked() ||
        !setup_context_locked(ctx) ||
        !prepare_counter_data_locked()) {
        state.mutex.unlock();
        return -1;
    }

    if (!driver_ok(driver.pfn_cuStreamSynchronize((CUstream) state.launch.stream), "cuStreamSynchronize(pre)") ||
        !driver_ok(driver.pfn_cuEventRecord(state.start_event, (CUstream) state.launch.stream), "cuEventRecord(start)")) {
        state.mutex.unlock();
        return -1;
    }
    state.active_events = true;

    if (!state.supported_metrics.empty()) {
        CUpti_PmSampling_Start_Params start_params =
            {CUpti_PmSampling_Start_Params_STRUCT_SIZE};
        start_params.pPmSamplingObject = state.pm_sampling;
        if (!cupti_ok(cuptiPmSamplingStart(&start_params), "cuptiPmSamplingStart")) {
            state.mutex.unlock();
            return -1;
        }
        state.active_pm_sampling = true;
    }

    state.active = true;
    return 0;
}

extern "C" void neutrino_plugin_end_launch_v1(
    const neutrino_plugin_launch_context_v1* context,
    CUresult launch_result) {
    (void) context;
    if (!state.active) {
        state.mutex.unlock();
        return;
    }

    bool end_event_recorded = false;
    if (state.active_events &&
        driver_ok(driver.pfn_cuEventRecord(state.end_event, (CUstream) state.launch.stream), "cuEventRecord(end)")) {
        end_event_recorded = true;
    }

    if (end_event_recorded &&
        driver_ok(driver.pfn_cuEventSynchronize(state.end_event), "cuEventSynchronize(end)")) {
        float elapsed_ms = 0.0f;
        if (driver_ok(driver.pfn_cuEventElapsedTime(&elapsed_ms, state.start_event, state.end_event),
                      "cuEventElapsedTime")) {
            state.event_duration_s = (double) elapsed_ms / 1000.0;
        }
    }

    if (state.active_pm_sampling) {
        CUpti_PmSampling_Stop_Params stop_params =
            {CUpti_PmSampling_Stop_Params_STRUCT_SIZE};
        stop_params.pPmSamplingObject = state.pm_sampling;
        if (cupti_ok(cuptiPmSamplingStop(&stop_params), "cuptiPmSamplingStop")) {
            evaluate_pm_metrics_locked();
        }
    }
    write_csv_row_locked(launch_result);
    state.active = false;
    state.active_pm_sampling = false;
    state.mutex.unlock();
}
