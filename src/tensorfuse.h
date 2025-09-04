/*
 * Copyright (c) 2024, TensorFuse Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * TensorFuse: Tensor-Core-Optimized Transformer Inference Library
 * 
 * Main API header providing high-level interface for fused transformer operations.
 * 
 * Key Features:
 * - Fused GEMM+Bias+GELU kernels with 2-7× speedups over cuBLASLt
 * - Fused Softmax+Dropout for optimized attention computation
 * - Architecture-specific optimizations (Ampere, Ada, Hopper)
 * - Support for FP32, FP16, BF16, and FP8 data types
 * - Automatic kernel tuning and caching
 * - Memory pool management for optimal performance
 */

#pragma once

#include "tensorfuse/types.h"
#include "tensorfuse/config.h"
#include "tensorfuse/profiler.h"

#include <cuda_runtime.h>

/**
 * @brief TensorFuse namespace containing all library APIs
 */
namespace tensorfuse {

/**
 * @brief Initialize TensorFuse runtime
 * @param device_id CUDA device ID (-1 for current device)
 * @return TensorFuseStatus indicating success or failure
 */
TensorFuseStatus initialize(int device_id = -1);

/**
 * @brief Cleanup TensorFuse runtime
 */
void cleanup();

/**
 * @brief Get the current runtime context
 * @return Reference to the global runtime context
 */
runtime::RuntimeContext& get_context();

/**
 * @brief Get device information
 * @return DeviceInfo structure containing device capabilities
 */
DeviceInfo get_device_info();

/**
 * @brief Set random seed for reproducible results
 * @param seed Random seed value
 * @return TensorFuseStatus indicating success or failure
 */
TensorFuseStatus set_random_seed(unsigned long long seed);

// ==============================================================================
// High-Level Fused Operations API
// ==============================================================================

/**
 * @brief Fused GEMM + Bias + GELU operation
 * 
 * Computes: output = GELU(A @ B + bias)
 * 
 * @param A Input matrix A [M, K]
 * @param B Input matrix B [K, N] 
 * @param bias Bias vector [N] (broadcasted)
 * @param output Output matrix [M, N]
 * @param M Number of rows in A and output
 * @param N Number of columns in B and output
 * @param K Inner dimension (columns of A, rows of B)
 * @param data_type Data type (FP32, FP16, BF16)
 * @param alpha Scaling factor for A@B
 * @param beta Scaling factor for bias (usually 1.0)
 * @param stream CUDA stream (nullptr for default stream)
 * @return TensorFuseStatus indicating success or failure
 */
TensorFuseStatus fused_gemm_bias_gelu(
    const void* A,
    const void* B, 
    const void* bias,
    void* output,
    int M, int N, int K,
    DataType data_type,
    float alpha = 1.0f,
    float beta = 1.0f,
    cudaStream_t stream = nullptr
);

/**
 * @brief Fused Softmax + Dropout operation
 * 
 * Computes: output = dropout(softmax(input))
 * 
 * @param input Input tensor [batch_size, num_heads, seq_len, seq_len]
 * @param output Output tensor [batch_size, num_heads, seq_len, seq_len]
 * @param dropout_mask Dropout mask [batch_size, num_heads, seq_len, seq_len]
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param num_heads Number of attention heads
 * @param head_dim Dimension per head
 * @param dropout_prob Dropout probability (0.0 to 1.0)
 * @param data_type Data type (FP32, FP16, BF16)
 * @param stream CUDA stream (nullptr for default stream)
 * @return TensorFuseStatus indicating success or failure
 */
TensorFuseStatus fused_softmax_dropout(
    const void* input,
    void* output,
    unsigned char* dropout_mask,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    float dropout_prob,
    DataType data_type,
    cudaStream_t stream = nullptr
);

// ==============================================================================
// Memory Management API
// ==============================================================================

/**
 * @brief Allocate device memory
 * @param size Size in bytes
 * @param alignment Memory alignment (default: 128 bytes)
 * @return Pointer to allocated memory or nullptr on failure
 */
void* allocate_device(size_t size, size_t alignment = 128);

/**
 * @brief Deallocate device memory
 * @param ptr Pointer to deallocate
 */
void deallocate_device(void* ptr);

/**
 * @brief Allocate host memory (pinned if possible)
 * @param size Size in bytes
 * @param alignment Memory alignment (default: 128 bytes)
 * @return Pointer to allocated memory or nullptr on failure
 */
void* allocate_host(size_t size, size_t alignment = 128);

/**
 * @brief Deallocate host memory
 * @param ptr Pointer to deallocate
 */
void deallocate_host(void* ptr);

/**
 * @brief Copy data from host to device
 * @param dst Device destination pointer
 * @param src Host source pointer
 * @param size Size in bytes
 * @param stream CUDA stream (nullptr for synchronous copy)
 * @return TensorFuseStatus indicating success or failure
 */
TensorFuseStatus copy_host_to_device(void* dst, const void* src, size_t size, cudaStream_t stream = nullptr);

/**
 * @brief Copy data from device to host
 * @param dst Host destination pointer
 * @param src Device source pointer
 * @param size Size in bytes
 * @param stream CUDA stream (nullptr for synchronous copy)
 * @return TensorFuseStatus indicating success or failure
 */
TensorFuseStatus copy_device_to_host(void* dst, const void* src, size_t size, cudaStream_t stream = nullptr);

/**
 * @brief Get memory usage statistics
 * @return MemoryStats structure containing usage information
 */
MemoryStats get_memory_stats();

// ==============================================================================
// Stream Management API
// ==============================================================================

/**
 * @brief Create a new CUDA stream
 * @return CUDA stream handle or nullptr on failure
 */
cudaStream_t create_stream();

/**
 * @brief Destroy a CUDA stream
 * @param stream Stream to destroy
 */
void destroy_stream(cudaStream_t stream);

/**
 * @brief Synchronize all operations
 * @return TensorFuseStatus indicating success or failure
 */
TensorFuseStatus synchronize();

/**
 * @brief Synchronize a specific stream
 * @param stream Stream to synchronize (nullptr for default stream)
 * @return TensorFuseStatus indicating success or failure
 */
TensorFuseStatus synchronize_stream(cudaStream_t stream = nullptr);

// ==============================================================================
// Utility Functions
// ==============================================================================

/**
 * @brief Get the version string
 * @return Version string in format "major.minor.patch"
 */
const char* get_version();

/**
 * @brief Get the build information
 * @return Build information string
 */
const char* get_build_info();

/**
 * @brief Convert error status to string
 * @param status TensorFuseStatus value
 * @return Human-readable error description
 */
const char* get_error_string(TensorFuseStatus status);

/**
 * @brief Check if the current GPU supports TensorFuse
 * @return true if supported, false otherwise
 */
bool is_gpu_supported();

/**
 * @brief Get supported data types for current GPU
 * @return Vector of supported DataType values
 */
std::vector<DataType> get_supported_data_types();

} // namespace tensorfuse

// ==============================================================================
// C API (for easier binding from other languages)
// ==============================================================================

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TENSORFUSE_SUCCESS = 0,
    TENSORFUSE_CUDA_ERROR,
    TENSORFUSE_CUTLASS_ERROR,
    TENSORFUSE_INVALID_ARGUMENT,
    TENSORFUSE_UNSUPPORTED_ARCH,
    TENSORFUSE_OUT_OF_MEMORY,
    TENSORFUSE_NOT_INITIALIZED
} tensorfuse_status_t;

typedef enum {
    TENSORFUSE_FP32 = 0,
    TENSORFUSE_FP16,
    TENSORFUSE_BF16,
    TENSORFUSE_FP8
} tensorfuse_data_type_t;

/**
 * @brief C API: Initialize TensorFuse
 */
tensorfuse_status_t tensorfuse_init(int device_id);

/**
 * @brief C API: Cleanup TensorFuse
 */
void tensorfuse_cleanup(void);

/**
 * @brief C API: Fused GEMM + Bias + GELU
 */
tensorfuse_status_t tensorfuse_fused_gemm_bias_gelu(
    const void* A, const void* B, const void* bias, void* output,
    int M, int N, int K,
    tensorfuse_data_type_t data_type,
    float alpha, float beta,
    void* stream  // cudaStream_t cast to void*
);

/**
 * @brief C API: Fused Softmax + Dropout
 */
tensorfuse_status_t tensorfuse_fused_softmax_dropout(
    const void* input, void* output, unsigned char* dropout_mask,
    int batch_size, int seq_len, int num_heads, int head_dim,
    float dropout_prob,
    tensorfuse_data_type_t data_type,
    void* stream  // cudaStream_t cast to void*
);

/**
 * @brief C API: Get error string
 */
const char* tensorfuse_get_error_string(tensorfuse_status_t status);

#ifdef __cplusplus
}
#endif 