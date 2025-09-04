#pragma once

/**
 * @file profiler.h
 * @brief Profiling interface for TensorFuse library
 */

#include "types.h"
#include <cuda_runtime.h>

#if TENSORFUSE_ENABLE_PROFILING
#include <cub/detail/nvtx3.hpp>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ==============================================================================
// Profiling Interface
// ==============================================================================

/**
 * @brief Initialize profiling system
 * @param config Profiling configuration
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_profiler_init(const TensorFuseProfileConfig* config);

/**
 * @brief Shutdown profiling system
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_profiler_shutdown();

/**
 * @brief Start profiling a region
 * @param name Region name
 * @param category Category for grouping
 * @return Region ID for ending the profile
 */
int tensorfuse_profiler_start_region(const char* name, const char* category);

/**
 * @brief End profiling a region
 * @param region_id Region ID returned by start_region
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_profiler_end_region(int region_id);

/**
 * @brief Record a kernel launch
 * @param kernel_name Kernel name
 * @param grid_dim Grid dimensions
 * @param block_dim Block dimensions
 * @param shared_mem_bytes Shared memory size
 * @param stream CUDA stream
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_profiler_record_kernel(
    const char* kernel_name,
    dim3 grid_dim,
    dim3 block_dim,
    size_t shared_mem_bytes,
    cudaStream_t stream
);

/**
 * @brief Record memory transfer
 * @param src_type Source memory type (host/device)
 * @param dst_type Destination memory type (host/device)
 * @param size_bytes Transfer size in bytes
 * @param stream CUDA stream
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_profiler_record_memcpy(
    const char* src_type,
    const char* dst_type,
    size_t size_bytes,
    cudaStream_t stream
);

/**
 * @brief Get current profiling metrics
 * @param metrics Output metrics structure
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_profiler_get_metrics(TensorFuseMetrics* metrics);

/**
 * @brief Reset profiling counters
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_profiler_reset();

/**
 * @brief Save profiling results to file
 * @param filename Output filename
 * @param format Output format ("json", "csv", "binary")
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_profiler_save(const char* filename, const char* format);

// ==============================================================================
// NVTX Integration
// ==============================================================================

#if TENSORFUSE_ENABLE_PROFILING

/**
 * @brief NVTX range handle for RAII in C++
 */
typedef struct {
    nvtxRangeId_t range_id;
    bool active;
} TensorFuseNvtxRange;

/**
 * @brief Start NVTX range
 * @param name Range name
 * @param color Color for visualization (0xRRGGBB)
 * @return NVTX range handle
 */
TensorFuseNvtxRange tensorfuse_nvtx_range_start(const char* name, uint32_t color);

/**
 * @brief End NVTX range
 * @param range NVTX range handle
 */
void tensorfuse_nvtx_range_end(TensorFuseNvtxRange* range);

/**
 * @brief Mark a point in execution
 * @param name Marker name
 * @param color Marker color (0xRRGGBB)
 */
void tensorfuse_nvtx_mark(const char* name, uint32_t color);

#else

// Dummy implementations when profiling is disabled  
typedef struct {
    int dummy;
    bool active;
} TensorFuseNvtxRange;

#define tensorfuse_nvtx_range_start(name, color) ((TensorFuseNvtxRange){0, false})
#define tensorfuse_nvtx_range_end(range) ((void)0)
#define tensorfuse_nvtx_mark(name, color) ((void)0)

#endif

// ==============================================================================
// Profiling Macros
// ==============================================================================

#if TENSORFUSE_ENABLE_PROFILING

/**
 * @brief Profile a code region
 */
#define TENSORFUSE_PROFILE_REGION(name, category) \
    int _profile_id = tensorfuse_profiler_start_region(name, category); \
    TensorFuseNvtxRange _nvtx_range = tensorfuse_nvtx_range_start(name, 0x00FF00FF); \
    __attribute__((cleanup(tensorfuse_profile_cleanup))) \
    struct { int id; TensorFuseNvtxRange* range; } _profile_data = {_profile_id, &_nvtx_range}

/**
 * @brief Profile a kernel launch
 */
#define TENSORFUSE_PROFILE_KERNEL(name, grid, block, smem, stream) \
    do { \
        tensorfuse_profiler_record_kernel(name, grid, block, smem, stream); \
        tensorfuse_nvtx_mark("Kernel: " name, 0xFF0000FF); \
    } while(0)

/**
 * @brief Profile memory transfer
 */
#define TENSORFUSE_PROFILE_MEMCPY(src, dst, size, stream) \
    do { \
        tensorfuse_profiler_record_memcpy(src, dst, size, stream); \
        tensorfuse_nvtx_mark("MemCpy: " src " -> " dst, 0x0000FFFF); \
    } while(0)

/**
 * @brief Cleanup function for profile region
 */
static inline void tensorfuse_profile_cleanup(void* data) {
    // Define a consistent struct type
    typedef struct {
        int id;
        TensorFuseNvtxRange* range;
    } ProfileCleanupData;
    
    ProfileCleanupData* profile_data = (ProfileCleanupData*)data;
    tensorfuse_profiler_end_region(profile_data->id);
    tensorfuse_nvtx_range_end(profile_data->range);
}

#else

// Dummy macros when profiling is disabled
#define TENSORFUSE_PROFILE_REGION(name, category) ((void)0)
#define TENSORFUSE_PROFILE_KERNEL(name, grid, block, smem, stream) ((void)0)
#define TENSORFUSE_PROFILE_MEMCPY(src, dst, size, stream) ((void)0)

#endif

// ==============================================================================
// Performance Timing Utilities
// ==============================================================================

/**
 * @brief High-resolution timer
 */
typedef struct {
    double start_time;
    double end_time;
    bool running;
} TensorFuseTimer;

/**
 * @brief Start timer
 * @param timer Timer instance
 */
void tensorfuse_timer_start(TensorFuseTimer* timer);

/**
 * @brief Stop timer
 * @param timer Timer instance
 */
void tensorfuse_timer_stop(TensorFuseTimer* timer);

/**
 * @brief Get elapsed time in milliseconds
 * @param timer Timer instance
 * @return Elapsed time in milliseconds
 */
double tensorfuse_timer_elapsed_ms(const TensorFuseTimer* timer);

/**
 * @brief CUDA event-based timer
 */
typedef struct {
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    bool events_created;
    bool timing_started;
} TensorFuseCudaTimer;

/**
 * @brief Create CUDA timer
 * @param timer Timer instance
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_cuda_timer_create(TensorFuseCudaTimer* timer);

/**
 * @brief Destroy CUDA timer
 * @param timer Timer instance
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_cuda_timer_destroy(TensorFuseCudaTimer* timer);

/**
 * @brief Start CUDA timing
 * @param timer Timer instance
 * @param stream CUDA stream
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_cuda_timer_start(TensorFuseCudaTimer* timer, cudaStream_t stream);

/**
 * @brief Stop CUDA timing
 * @param timer Timer instance
 * @param stream CUDA stream
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_cuda_timer_stop(TensorFuseCudaTimer* timer, cudaStream_t stream);

/**
 * @brief Get elapsed CUDA time in milliseconds
 * @param timer Timer instance
 * @param elapsed_ms Output elapsed time
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_cuda_timer_elapsed_ms(TensorFuseCudaTimer* timer, float* elapsed_ms);

// ==============================================================================
// Roofline Model Support
// ==============================================================================

/**
 * @brief Roofline model parameters
 */
typedef struct {
    float peak_flops;                    ///< Peak FLOPS (ops/sec)
    float peak_bandwidth;                ///< Peak memory bandwidth (bytes/sec)
    float arithmetic_intensity;          ///< Arithmetic intensity (ops/byte)
    float achieved_flops;                ///< Achieved FLOPS (ops/sec)
    float achieved_bandwidth;            ///< Achieved bandwidth (bytes/sec)
    float efficiency_flops;              ///< FLOPS efficiency (%)
    float efficiency_bandwidth;          ///< Bandwidth efficiency (%)
    bool is_compute_bound;               ///< Whether operation is compute-bound
    bool is_memory_bound;                ///< Whether operation is memory-bound
} TensorFuseRooflineModel;

/**
 * @brief Calculate roofline model metrics
 * @param ops_count Number of operations
 * @param bytes_transferred Number of bytes transferred
 * @param time_ms Execution time in milliseconds
 * @param model Output roofline model
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_calculate_roofline(
    size_t ops_count,
    size_t bytes_transferred,
    float time_ms,
    TensorFuseRooflineModel* model
);

#ifdef __cplusplus
}

// ==============================================================================
// C++ RAII Wrappers
// ==============================================================================

#ifdef TENSORFUSE_ENABLE_PROFILING

namespace tensorfuse {

/**
 * @brief RAII wrapper for profiling regions
 */
class ProfileRegion {
public:
    ProfileRegion(const char* name, const char* category = "Default") {
        region_id_ = tensorfuse_profiler_start_region(name, category);
        nvtx_range_ = tensorfuse_nvtx_range_start(name, 0x00FF00FF);
    }
    
    ~ProfileRegion() {
        tensorfuse_profiler_end_region(region_id_);
        tensorfuse_nvtx_range_end(&nvtx_range_);
    }
    
    // Non-copyable
    ProfileRegion(const ProfileRegion&) = delete;
    ProfileRegion& operator=(const ProfileRegion&) = delete;
    
private:
    int region_id_;
    TensorFuseNvtxRange nvtx_range_;
};

/**
 * @brief RAII wrapper for CUDA timers
 */
class CudaTimer {
public:
    CudaTimer() {
        tensorfuse_cuda_timer_create(&timer_);
    }
    
    ~CudaTimer() {
        tensorfuse_cuda_timer_destroy(&timer_);
    }
    
    void start(cudaStream_t stream = 0) {
        tensorfuse_cuda_timer_start(&timer_, stream);
    }
    
    void stop(cudaStream_t stream = 0) {
        tensorfuse_cuda_timer_stop(&timer_, stream);
    }
    
    float elapsed_ms() {
        float elapsed;
        tensorfuse_cuda_timer_elapsed_ms(&timer_, &elapsed);
        return elapsed;
    }
    
    // Non-copyable
    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;
    
private:
    TensorFuseCudaTimer timer_;
};

} // namespace tensorfuse

// Convenience macro for C++ profiling
#define TENSORFUSE_PROFILE_SCOPE(name) \
    tensorfuse::ProfileRegion _profile_region(name)

#define TENSORFUSE_PROFILE_SCOPE_CAT(name, category) \
    tensorfuse::ProfileRegion _profile_region(name, category)

#endif // TENSORFUSE_ENABLE_PROFILING

#endif // __cplusplus 