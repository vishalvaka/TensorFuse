/*
 * Copyright (c) 2024, TensorFuse Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Memory management system for CUDA memory allocation and pooling
 */

#pragma once

#include "tensorfuse/types.h"
#include <cuda_runtime.h>
#include <memory>
#include <mutex>

namespace tensorfuse {
namespace runtime {

/**
 * @brief Memory usage statistics
 */
struct MemoryStats {
    size_t total_allocated;  ///< Total memory allocated
    size_t free_memory;      ///< Free memory available
    size_t used_memory;      ///< Currently used memory
    size_t peak_usage;       ///< Peak memory usage
    size_t num_blocks;       ///< Number of memory blocks
};

/**
 * @brief Memory manager for CUDA memory allocation and pooling
 */
class MemoryManager {
public:
    MemoryManager();
    ~MemoryManager();

    // Non-copyable
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;

    /**
     * @brief Allocate device memory
     * @param size Size in bytes
     * @param alignment Memory alignment (default: 128 bytes)
     * @return Pointer to allocated memory or nullptr on failure
     */
    void* allocate(size_t size, size_t alignment = 128);

    /**
     * @brief Deallocate device memory
     * @param ptr Pointer to deallocate
     */
    void deallocate(void* ptr);

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
     * @brief Copy data from device to device
     * @param dst Device destination pointer
     * @param src Device source pointer
     * @param size Size in bytes
     * @param stream CUDA stream (nullptr for synchronous copy)
     * @return TensorFuseStatus indicating success or failure
     */
    TensorFuseStatus copy_device_to_device(void* dst, const void* src, size_t size, cudaStream_t stream = nullptr);

    /**
     * @brief Set device memory to a value
     * @param ptr Device pointer
     * @param value Value to set
     * @param size Size in bytes
     * @param stream CUDA stream (nullptr for synchronous operation)
     * @return TensorFuseStatus indicating success or failure
     */
    TensorFuseStatus memset_device(void* ptr, int value, size_t size, cudaStream_t stream = nullptr);

    /**
     * @brief Get memory usage statistics
     * @return MemoryStats structure containing usage information
     */
    MemoryStats get_memory_stats() const;

    /**
     * @brief Clear memory pool
     */
    void clear_memory_pool();

    /**
     * @brief Enable or disable memory pool
     * @param enabled true to enable, false to disable
     */
    void set_memory_pool_enabled(bool enabled);

    /**
     * @brief Check if memory pool is enabled
     * @return true if enabled
     */
    bool is_memory_pool_enabled() const;

private:
    /**
     * @brief Align size to specified boundary
     * @param size Size to align
     * @param alignment Alignment boundary
     * @return Aligned size
     */
    static size_t align_size(size_t size, size_t alignment);

    bool use_memory_pool_;
};

/**
 * @brief Initialize global memory manager
 */
void initialize_memory_manager();

/**
 * @brief Get global memory manager instance
 * @return Reference to global memory manager
 */
MemoryManager& get_memory_manager();

} // namespace runtime
} // namespace tensorfuse 