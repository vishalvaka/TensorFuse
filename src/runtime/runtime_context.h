/*
 * Copyright (c) 2024, TensorFuse Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Runtime context for managing CUDA streams, device state, and execution environment
 */

#pragma once

#include "tensorfuse/types.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <vector>
#include <memory>
#include <mutex>
#include <algorithm>

namespace tensorfuse {
namespace runtime {

// Forward declarations
class MemoryManager;

/**
 * @brief Runtime context managing CUDA streams, device state, and execution environment
 */
class RuntimeContext {
public:
    RuntimeContext();
    ~RuntimeContext();

    // Non-copyable
    RuntimeContext(const RuntimeContext&) = delete;
    RuntimeContext& operator=(const RuntimeContext&) = delete;

    /**
     * @brief Initialize the runtime context
     * @param device_id CUDA device ID (-1 for current device)
     * @return TensorFuseStatus indicating success or failure
     */
    TensorFuseStatus initialize(int device_id = -1);

    /**
     * @brief Cleanup the runtime context
     */
    void cleanup();

    /**
     * @brief Get the main CUDA stream
     * @return Main CUDA stream handle
     */
    cudaStream_t get_main_stream() const;

    /**
     * @brief Create a new CUDA stream
     * @return New CUDA stream handle or nullptr on failure
     */
    cudaStream_t create_stream();

    /**
     * @brief Destroy a CUDA stream
     * @param stream Stream to destroy
     */
    void destroy_stream(cudaStream_t stream);

    /**
     * @brief Synchronize all operations on all streams
     * @return TensorFuseStatus indicating success or failure
     */
    TensorFuseStatus synchronize();

    /**
     * @brief Synchronize operations on a specific stream
     * @param stream Stream to synchronize (nullptr for main stream)
     * @return TensorFuseStatus indicating success or failure
     */
    TensorFuseStatus synchronize_stream(cudaStream_t stream = nullptr);

    /**
     * @brief Get the current device ID
     * @return CUDA device ID
     */
    int get_device_id() const;

    /**
     * @brief Get device properties
     * @return Reference to device properties
     */
    const cudaDeviceProp& get_device_properties() const;

    /**
     * @brief Get compute capability major version
     * @return Major version number
     */
    int get_compute_capability_major() const;

    /**
     * @brief Get compute capability minor version
     * @return Minor version number
     */
    int get_compute_capability_minor() const;

    /**
     * @brief Check if device supports Tensor Cores
     * @return true if Tensor Cores are supported
     */
    bool supports_tensor_cores() const;

    /**
     * @brief Check if device supports FP16
     * @return true if FP16 is supported
     */
    bool supports_fp16() const;

    /**
     * @brief Check if device supports BF16
     * @return true if BF16 is supported
     */
    bool supports_bf16() const;

    /**
     * @brief Check if device supports FP8
     * @return true if FP8 is supported
     */
    bool supports_fp8() const;

    /**
     * @brief Get maximum shared memory per block
     * @return Maximum shared memory in bytes
     */
    size_t get_max_shared_memory_per_block() const;

    /**
     * @brief Get maximum threads per block
     * @return Maximum threads per block
     */
    int get_max_threads_per_block() const;

    /**
     * @brief Get number of multiprocessors
     * @return Number of multiprocessors
     */
    int get_multiprocessor_count() const;

    /**
     * @brief Get total global memory
     * @return Total global memory in bytes
     */
    size_t get_total_global_memory() const;

    /**
     * @brief Get cuBLAS handle
     * @return cuBLAS handle
     */
    cublasHandle_t get_cublas_handle() const;

    /**
     * @brief Get cuDNN handle
     * @return cuDNN handle
     */
    cudnnHandle_t get_cudnn_handle() const;

    /**
     * @brief Get cuRAND generator
     * @return cuRAND generator
     */
    curandGenerator_t get_curand_generator() const;

    /**
     * @brief Get memory manager
     * @return Reference to memory manager
     */
    MemoryManager& get_memory_manager() const;

    /**
     * @brief Get device information
     * @return Device information structure
     */
    DeviceInfo get_device_info() const;

    /**
     * @brief Set random seed
     * @param seed Random seed value
     * @return TensorFuseStatus indicating success or failure
     */
    TensorFuseStatus set_random_seed(unsigned long long seed);

    /**
     * @brief Check if context is initialized
     * @return true if initialized
     */
    bool is_initialized() const;

private:
    int device_id_;
    int compute_capability_major_;
    int compute_capability_minor_;
    cudaDeviceProp device_properties_;
    
    cudaStream_t main_stream_;
    std::vector<cudaStream_t> additional_streams_;
    
    cublasHandle_t cublas_handle_;
    cudnnHandle_t cudnn_handle_;
    curandGenerator_t curand_generator_;
    
    MemoryManager* memory_manager_;
    bool is_initialized_;
};

/**
 * @brief Initialize global runtime context
 * @param device_id CUDA device ID (-1 for current device)
 * @return TensorFuseStatus indicating success or failure
 */
TensorFuseStatus initialize_runtime_context(int device_id = -1);

/**
 * @brief Get global runtime context
 * @return Reference to global runtime context
 */
RuntimeContext& get_runtime_context();

/**
 * @brief Cleanup global runtime context
 */
void cleanup_runtime_context();

} // namespace runtime
} // namespace tensorfuse 