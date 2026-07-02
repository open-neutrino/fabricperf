/*
 * Shared FabricPerf plugin helpers.
 *
 * This header is intentionally plugin-local: it assumes each FabricPerf source
 * owns a file-scope `api` pointer set by neutrino_plugin_init_v1().
 */
#ifndef FABRICPERF_COMMON_H
#define FABRICPERF_COMMON_H

#include "plugin.h"

#include <dlfcn.h>
#ifndef FABRICPERF_NO_MPI
#include <mpi.h>
#endif
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Stable log label; copied plugins can override this before including common.h. */
#ifndef PLUGIN_NAME
#define PLUGIN_NAME "fabricperf"
#endif

/* FabricPerf buffer geometry shared with the injected probe snippets. */
#define FABRICPERF_LEADER_SPACES 3
#define FABRICPERF_FOLLOWER_SPACES 5
#define FABRICPERF_PTP_SAMPLES_PER_LEADER 20
/* Latency PTP direct all-pairs calibration is bounded to 2..8 ranks. */
#define FABRICPERF_MAX_PTP_RANKS 8
/* Diagnostic PTP rows for every ordered non-self pair at max scale. */
#define FABRICPERF_PTP_DIAGNOSTIC_RECORDS \
    (FABRICPERF_MAX_PTP_RANKS * (FABRICPERF_MAX_PTP_RANKS - 1) * FABRICPERF_PTP_SAMPLES_PER_LEADER)
#define FABRICPERF_SENDRECV_MESSAGE_SLOTS 4096
/* Latency mailbox slots per source rank; example: slot[srcPeer][vblock]. */
#define FABRICPERF_LATENCY_MAILBOX_VBLOCKS 4096
/* Latency mailbox tag kind reused from the PTP message low bits. */
#define FABRICPERF_LATENCY_MAILBOX_KIND 160
#define FABRICPERF_MAX_CHANNEL 16
/* Result scratch stores compact PTP diagnostics; example: 8 * 7 * 20 rows. */
#define FABRICPERF_RESULT_SLOTS FABRICPERF_PTP_DIAGNOSTIC_RECORDS
/* Throughput mode hard-codes this many logical receive workgroups per CTA. */
#define FABRICPERF_THROUGHPUT_WORKGROUPS 4
/* Each throughput workgroup owns this many u32 cells in the CTA stream. */
#define FABRICPERF_THROUGHPUT_CELLS_PER_WORKGROUP 1024
/* The first cells store duration and total arrivals before timestamp deltas. */
#define FABRICPERF_THROUGHPUT_HEADER_CELLS 2
/* Total u32 cells in the logical CTA throughput stream. */
#define FABRICPERF_THROUGHPUT_SLOTS \
    (FABRICPERF_THROUGHPUT_WORKGROUPS * FABRICPERF_THROUGHPUT_CELLS_PER_WORKGROUP)
/* Throughput analysis bins timestamp deltas at this nanosecond-scale width. */
#define FABRICPERF_THROUGHPUT_BIN_NS 8000
/* Throughput mode splits each saved-record stream into this many partitions. */
#define FABRICPERF_THROUGHPUT_PARTITIONS_DEFAULT 4

/* One FabricPerf message/result cell. PTX stores these as v2.u64 pairs. */
typedef struct fabricperf_slot {
    uint64_t x; /* Tag, timestamp, or result component depending on probe path. */
    uint64_t y; /* Companion timestamp/result component for the same slot. */
} fabricperf_slot_t;

#ifdef __cplusplus
/* C++ plugin code uses the historical type name; keep it as an alias. */
using FabricPerfSlot = fabricperf_slot_t;
#endif

/* Log one printf-style plugin diagnostic through Neutrino's event.log API. */
#define PLUGIN_LOG(...)                                                               \
    do {                                                                              \
        if (api != NULL && api->abi_version == NEUTRINO_PLUGIN_ABI_VERSION &&          \
            api->api_size >= offsetof(neutrino_plugin_api_v1, log) + sizeof(api->log) && \
            api->log != NULL) {                                                       \
            api->log(PLUGIN_NAME, __VA_ARGS__);                                       \
        }                                                                             \
    } while (0)

/* Return one failure value after recording the diagnostic in event.log. */
#define PLUGIN_FAIL(fail_ret, ...)          \
    do {                                    \
        PLUGIN_LOG(__VA_ARGS__);            \
        return (fail_ret);                  \
    } while (0)

/* Load one CUDA symbol through the plugin-owned real CUDA driver handle. */
#define PLUGIN_LOAD_CUDA_SYMBOL(table, field, symbol_name, required, fail_ret)     \
    do {                                                                          \
        dlerror();                                                                 \
        *(void**) (&((table)->field)) = dlsym((table)->handle, (symbol_name));      \
        const char* fabricperf_load_error = dlerror();                             \
        if ((required) && (((table)->field) == NULL || fabricperf_load_error != NULL)) { \
            PLUGIN_LOG("missing CUDA symbol %s: %s",                               \
                       (symbol_name),                                              \
                       fabricperf_load_error != NULL ? fabricperf_load_error : "not found"); \
            return (fail_ret);                                                     \
        }                                                                          \
    } while (0)

/* Validate the Neutrino plugin ABI before init continues. */
#define PLUGIN_REQUIRE_API(fail_ret)                                               \
    do {                                                                          \
        if (api == NULL || api->abi_version != NEUTRINO_PLUGIN_ABI_VERSION) {       \
            return (fail_ret);                                                     \
        }                                                                          \
        if (api->api_size < sizeof(neutrino_plugin_api_v1)) {                      \
            PLUGIN_LOG("loaded with an incomplete Neutrino plugin API.");           \
            return (fail_ret);                                                     \
        }                                                                          \
    } while (0)

/* Align one size up to CUDA granularity; zero granularity leaves the value unchanged. */
#define PLUGIN_ALIGN_UP(value, granularity) \
    ((granularity) == 0 ? (value) : ((((value) + (granularity) - 1) / (granularity)) * (granularity)))

/* Null-safe symbol-name compare used by FabricPerf resolve-symbol dispatch. */
#define PLUGIN_IS_SYMBOL(name, target) \
    ((name) != NULL && (target) != NULL && strcmp((name), (target)) == 0)

/* Convert one CUDA result into a plugin status using the local api pointer. */
#define PLUGIN_CHECK_CUDA(result, call)                                          \
    ({                                                                          \
        CUresult fabricperf_cuda_result = (result);                              \
        int fabricperf_cuda_status = 0;                                          \
        if (fabricperf_cuda_result != CUDA_SUCCESS) {                            \
            PLUGIN_LOG("CUDA error from %s: %d",                                 \
                       (call) != NULL ? (call) : "(unknown)",                    \
                       (int) fabricperf_cuda_result);                            \
            fabricperf_cuda_status = -1;                                         \
        }                                                                        \
        fabricperf_cuda_status;                                                  \
    })

#ifndef FABRICPERF_NO_MPI
/* Log one MPI failure with a consistent FabricPerf event.log message. */
#define PLUGIN_LOG_MPI_ERROR(status_expr, call)                                          \
    do {                                                                                 \
        int fabricperf_mpi_error_code = (status_expr);                                   \
        char fabricperf_mpi_text[MPI_MAX_ERROR_STRING] = {0};                            \
        int fabricperf_mpi_len = 0;                                                      \
        if (MPI_Error_string(fabricperf_mpi_error_code,                                  \
                             fabricperf_mpi_text,                                       \
                             &fabricperf_mpi_len) == MPI_SUCCESS) {                      \
            if (fabricperf_mpi_len >= 0 &&                                               \
                (size_t) fabricperf_mpi_len < sizeof(fabricperf_mpi_text)) {             \
                fabricperf_mpi_text[fabricperf_mpi_len] = '\0';                          \
            } else {                                                                      \
                fabricperf_mpi_text[sizeof(fabricperf_mpi_text) - 1] = '\0';             \
            }                                                                             \
            PLUGIN_LOG("MPI error from %s: %s", (call), fabricperf_mpi_text);             \
        } else {                                                                          \
            PLUGIN_LOG("MPI error from %s: MPI error %d",                                \
                       (call),                                                           \
                       fabricperf_mpi_error_code);                                       \
        }                                                                                 \
    } while (0)

/* Return fail_ret when one MPI call fails. */
#define PLUGIN_CHECK_MPI(mpi_expr, call, fail_ret)           \
    do {                                                     \
        int fabricperf_mpi_checked_status = (mpi_expr);      \
        if (fabricperf_mpi_checked_status != MPI_SUCCESS) {  \
            PLUGIN_LOG_MPI_ERROR(fabricperf_mpi_checked_status, call); \
            return (fail_ret);                               \
        }                                                    \
    } while (0)
#endif

#endif
