#pragma once

/**
 * @file config.h
 * @brief Configuration constants and macros for TensorFuse library
 */

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// ==============================================================================
// Version Information
// ==============================================================================

#define TENSORFUSE_VERSION_MAJOR 1
#define TENSORFUSE_VERSION_MINOR 0
#define TENSORFUSE_VERSION_PATCH 0
#define TENSORFUSE_VERSION_STRING "1.0.0"

// ==============================================================================
// Build Configuration
// ==============================================================================

// Enable FP8 support (requires Hopper+ architecture)
#ifndef TENSORFUSE_ENABLE_FP8
#define TENSORFUSE_ENABLE_FP8 0
#endif

// Enable profiling support
#ifndef TENSORFUSE_ENABLE_PROFILING
#define TENSORFUSE_ENABLE_PROFILING 1
#endif

// Enable debug mode
#ifndef TENSORFUSE_DEBUG
#define TENSORFUSE_DEBUG 0
#endif

// Enable assertions
#ifndef TENSORFUSE_ENABLE_ASSERTIONS
#define TENSORFUSE_ENABLE_ASSERTIONS 1
#endif

// ==============================================================================
// Hardware Configuration
// ==============================================================================

// Minimum required CUDA compute capability
#define TENSORFUSE_MIN_COMPUTE_CAPABILITY_MAJOR 8
#define TENSORFUSE_MIN_COMPUTE_CAPABILITY_MINOR 0

// Supported CUDA architectures
#define TENSORFUSE_CUDA_ARCH_AMPERE   80  // A100, A30, A10, RTX 30xx series
#define TENSORFUSE_CUDA_ARCH_ADA      89  // RTX 40xx series, L40S
#define TENSORFUSE_CUDA_ARCH_HOPPER   90  // H100, H200

// Warp size
#define TENSORFUSE_WARP_SIZE 32

// Maximum number of threads per block
#define TENSORFUSE_MAX_THREADS_PER_BLOCK 1024

// Maximum shared memory per block (bytes)
#define TENSORFUSE_MAX_SHARED_MEMORY_PER_BLOCK (48 * 1024)

// ==============================================================================
// Memory Configuration
// ==============================================================================

// Memory alignment for tensors (bytes)
#define TENSORFUSE_MEMORY_ALIGNMENT 128

// Default workspace size per device (bytes)
#define TENSORFUSE_DEFAULT_WORKSPACE_SIZE (256 * 1024 * 1024)  // 256 MB

// Maximum number of CUDA devices supported
#define TENSORFUSE_MAX_DEVICES 16

// ==============================================================================
// Kernel Configuration
// ==============================================================================

// Default tile sizes for GEMM operations
#define TENSORFUSE_DEFAULT_TILE_M 128
#define TENSORFUSE_DEFAULT_TILE_N 128
#define TENSORFUSE_DEFAULT_TILE_K 64

// Default warp tile sizes
#define TENSORFUSE_DEFAULT_WARP_M 64
#define TENSORFUSE_DEFAULT_WARP_N 64
#define TENSORFUSE_DEFAULT_WARP_K 16

// Default number of pipeline stages
#define TENSORFUSE_DEFAULT_STAGES 3

// Maximum number of kernel configurations to cache
#define TENSORFUSE_MAX_KERNEL_CACHE_SIZE 1024

// ==============================================================================
// Autotuner Configuration
// ==============================================================================

// Default autotuner settings
#define TENSORFUSE_DEFAULT_AUTOTUNE_ITERATIONS 100
#define TENSORFUSE_DEFAULT_AUTOTUNE_TIME_LIMIT 300.0f  // 5 minutes
#define TENSORFUSE_DEFAULT_WARMUP_RUNS 3
#define TENSORFUSE_DEFAULT_TIMING_RUNS 10
#define TENSORFUSE_DEFAULT_RELATIVE_THRESHOLD 0.05f  // 5%

// Maximum search space size for autotuning
#define TENSORFUSE_MAX_AUTOTUNE_SEARCH_SPACE 10000

// ==============================================================================
// Profiling Configuration
// ==============================================================================

// Default profiling settings
#define TENSORFUSE_DEFAULT_PROFILE_WARMUP 10
#define TENSORFUSE_DEFAULT_PROFILE_ITERATIONS 100

// Maximum number of profiling events
#define TENSORFUSE_MAX_PROFILE_EVENTS 10000

// ==============================================================================
// Numerical Constants
// ==============================================================================

// Epsilon for layer normalization
#define TENSORFUSE_LAYERNORM_EPS 1e-5f

// Epsilon for numerical stability
#define TENSORFUSE_NUMERICAL_EPS 1e-8f

// Maximum value for softmax stability
#define TENSORFUSE_SOFTMAX_MAX_VALUE 85.0f

// FP8 scaling constants
#define TENSORFUSE_FP8_E4M3_MAX 448.0f
#define TENSORFUSE_FP8_E5M2_MAX 57344.0f

// ==============================================================================
// Error Handling
// ==============================================================================

// Error checking macro
#if TENSORFUSE_ENABLE_ASSERTIONS
#define TENSORFUSE_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "TensorFuse assertion failed: %s at %s:%d\n", \
                    message, __FILE__, __LINE__); \
            abort(); \
        } \
    } while(0)
#else
#define TENSORFUSE_ASSERT(condition, message) ((void)0)
#endif

// CUDA error checking
#define TENSORFUSE_CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            return TENSORFUSE_ERROR_CUDA_ERROR; \
        } \
    } while(0)

// CUDA error checking with custom return value
#define TENSORFUSE_CUDA_CHECK_RETURN(call, retval) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            return retval; \
        } \
    } while(0)

// ==============================================================================
// Logging Configuration
// ==============================================================================

// Logging levels
#define TENSORFUSE_LOG_LEVEL_QUIET 0
#define TENSORFUSE_LOG_LEVEL_INFO 1
#define TENSORFUSE_LOG_LEVEL_DEBUG 2

// Default logging level
#ifndef TENSORFUSE_LOG_LEVEL
#define TENSORFUSE_LOG_LEVEL TENSORFUSE_LOG_LEVEL_INFO
#endif

// Logging macros
#if TENSORFUSE_LOG_LEVEL >= TENSORFUSE_LOG_LEVEL_INFO
#define TENSORFUSE_LOG_INFO(format, ...) \
    fprintf(stdout, "[TensorFuse INFO] " format "\n", ##__VA_ARGS__)
#else
#define TENSORFUSE_LOG_INFO(format, ...) ((void)0)
#endif

#if TENSORFUSE_LOG_LEVEL >= TENSORFUSE_LOG_LEVEL_DEBUG
#define TENSORFUSE_LOG_DEBUG(format, ...) \
    fprintf(stdout, "[TensorFuse DEBUG] " format "\n", ##__VA_ARGS__)
#else
#define TENSORFUSE_LOG_DEBUG(format, ...) ((void)0)
#endif

// ==============================================================================
// Platform-specific Configuration
// ==============================================================================

// Force inline for performance-critical functions
#ifdef __CUDACC__
#define TENSORFUSE_INLINE __forceinline__
#define TENSORFUSE_DEVICE __device__
#define TENSORFUSE_HOST __host__
#define TENSORFUSE_GLOBAL __global__
#else
#define TENSORFUSE_INLINE inline
#define TENSORFUSE_DEVICE
#define TENSORFUSE_HOST
#define TENSORFUSE_GLOBAL
#endif

// Memory space qualifiers
#define TENSORFUSE_SHARED __shared__
#define TENSORFUSE_CONSTANT __constant__

// ==============================================================================
// Feature Detection
// ==============================================================================

// Check for CUDA architecture support
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define TENSORFUSE_ARCH_AMPERE_PLUS 1
#else
#define TENSORFUSE_ARCH_AMPERE_PLUS 0
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
#define TENSORFUSE_ARCH_ADA_PLUS 1
#else
#define TENSORFUSE_ARCH_ADA_PLUS 0
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define TENSORFUSE_ARCH_HOPPER_PLUS 1
#else
#define TENSORFUSE_ARCH_HOPPER_PLUS 0
#endif

// FP8 support detection
#if TENSORFUSE_ENABLE_FP8 && TENSORFUSE_ARCH_HOPPER_PLUS
#define TENSORFUSE_HAS_FP8_SUPPORT 1
#else
#define TENSORFUSE_HAS_FP8_SUPPORT 0
#endif

#ifdef __cplusplus
}
#endif 