/*
 * Copyright (c) 2024, TensorFuse Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Memory management system for CUDA memory allocation and pooling
 */

#include "runtime/memory_manager.h"
#include "utils/error_handling.h"
#include "utils/cuda_utils.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>

namespace tensorfuse {
namespace runtime {

// Memory pool implementation
class MemoryPool {
private:
    struct Block {
        void* ptr;
        size_t size;
        bool is_free;
        
        Block(void* p, size_t s) : ptr(p), size(s), is_free(true) {}
    };
    
    std::vector<Block> blocks_;
    size_t total_allocated_;
    size_t peak_usage_;
    mutable std::mutex mutex_;
    
public:
    MemoryPool() : total_allocated_(0), peak_usage_(0) {}
    
    ~MemoryPool() {
        clear();
    }
    
    void* allocate(size_t size, size_t alignment = 128) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Align size to specified boundary
        size_t aligned_size = align_size(size, alignment);
        
        // Find a free block that can accommodate the request
        for (auto& block : blocks_) {
            if (block.is_free && block.size >= aligned_size) {
                block.is_free = false;
                
                // Split block if it's significantly larger
                if (block.size > aligned_size * 2) {
                    Block new_block(static_cast<char*>(block.ptr) + aligned_size,
                                   block.size - aligned_size);
                    blocks_.push_back(new_block);
                    block.size = aligned_size;
                }
                
                return block.ptr;
            }
        }
        
        // No suitable block found, allocate new one
        void* ptr = nullptr;
        cudaError_t error = cudaMalloc(&ptr, aligned_size);
        if (error != cudaSuccess) {
            return nullptr;
        }
        
        blocks_.emplace_back(ptr, aligned_size);
        blocks_.back().is_free = false;
        total_allocated_ += aligned_size;
        peak_usage_ = std::max(peak_usage_, total_allocated_);
        
        return ptr;
    }
    
    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (auto& block : blocks_) {
            if (block.ptr == ptr) {
                block.is_free = true;
                coalesce_free_blocks();
                return;
            }
        }
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (const auto& block : blocks_) {
            cudaFree(block.ptr);
        }
        blocks_.clear();
        total_allocated_ = 0;
    }
    
    size_t get_total_allocated() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return total_allocated_;
    }
    
    size_t get_peak_usage() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return peak_usage_;
    }
    
    MemoryStats get_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        MemoryStats stats{};
        stats.total_allocated = total_allocated_;
        stats.peak_usage = peak_usage_;
        stats.num_blocks = blocks_.size();
        
        size_t free_memory = 0;
        for (const auto& block : blocks_) {
            if (block.is_free) {
                free_memory += block.size;
            }
        }
        stats.free_memory = free_memory;
        stats.used_memory = total_allocated_ - free_memory;
        
        return stats;
    }
    
private:
    size_t align_size(size_t size, size_t alignment) {
        return ((size + alignment - 1) / alignment) * alignment;
    }
    
    void coalesce_free_blocks() {
        // Sort blocks by address
        std::sort(blocks_.begin(), blocks_.end(),
                  [](const Block& a, const Block& b) { return a.ptr < b.ptr; });
        
        // Merge adjacent free blocks
        for (size_t i = 0; i < blocks_.size() - 1; ) {
            if (blocks_[i].is_free && blocks_[i + 1].is_free &&
                static_cast<char*>(blocks_[i].ptr) + blocks_[i].size == blocks_[i + 1].ptr) {
                blocks_[i].size += blocks_[i + 1].size;
                blocks_.erase(blocks_.begin() + i + 1);
            } else {
                ++i;
            }
        }
    }
};

// Global memory pool instance
static MemoryPool* global_pool = nullptr;
static std::once_flag pool_init_flag;

void initialize_memory_pool() {
    std::call_once(pool_init_flag, []() {
        global_pool = new MemoryPool();
        
        // Register cleanup at exit
        std::atexit([]() {
            delete global_pool;
            global_pool = nullptr;
        });
    });
}

// MemoryManager implementation
MemoryManager::MemoryManager() : use_memory_pool_(true) {
    initialize_memory_pool();
}

MemoryManager::~MemoryManager() = default;

void* MemoryManager::allocate(size_t size, size_t alignment) {
    if (size == 0) return nullptr;
    
    if (use_memory_pool_ && global_pool) {
        return global_pool->allocate(size, alignment);
    } else {
        // Direct CUDA allocation
        void* ptr = nullptr;
        size_t aligned_size = align_size(size, alignment);
        cudaError_t error = cudaMalloc(&ptr, aligned_size);
        return (error == cudaSuccess) ? ptr : nullptr;
    }
}

void MemoryManager::deallocate(void* ptr) {
    if (!ptr) return;
    
    if (use_memory_pool_ && global_pool) {
        global_pool->deallocate(ptr);
    } else {
        cudaFree(ptr);
    }
}

void* MemoryManager::allocate_host(size_t size, size_t alignment) {
    if (size == 0) return nullptr;
    
    size_t aligned_size = align_size(size, alignment);
    void* ptr = nullptr;
    
    // Try to allocate pinned memory for better transfer performance
    cudaError_t error = cudaMallocHost(&ptr, aligned_size);
    if (error == cudaSuccess) {
        return ptr;
    }
    
    // Fall back to regular aligned allocation
    return aligned_alloc(alignment, aligned_size);
}

void MemoryManager::deallocate_host(void* ptr) {
    if (!ptr) return;
    
    // Check if it's pinned memory
    cudaPointerAttributes attrs;
    cudaError_t error = cudaPointerGetAttributes(&attrs, ptr);
    
    if (error == cudaSuccess && attrs.type == cudaMemoryTypeHost) {
        cudaFreeHost(ptr);
    } else {
        free(ptr);
    }
}

TensorFuseStatus MemoryManager::copy_host_to_device(void* dst, const void* src, size_t size, cudaStream_t stream) {
    cudaError_t error;
    
    if (stream) {
        error = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
    } else {
        error = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    }
    
    return (error == cudaSuccess) ? TENSORFUSE_SUCCESS : TENSORFUSE_ERROR_CUDA_ERROR;
}

TensorFuseStatus MemoryManager::copy_device_to_host(void* dst, const void* src, size_t size, cudaStream_t stream) {
    cudaError_t error;
    
    if (stream) {
        error = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
    } else {
        error = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    }
    
    return (error == cudaSuccess) ? TENSORFUSE_SUCCESS : TENSORFUSE_ERROR_CUDA_ERROR;
}

TensorFuseStatus MemoryManager::copy_device_to_device(void* dst, const void* src, size_t size, cudaStream_t stream) {
    cudaError_t error;
    
    if (stream) {
        error = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
    } else {
        error = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    }
    
    return (error == cudaSuccess) ? TENSORFUSE_SUCCESS : TENSORFUSE_ERROR_CUDA_ERROR;
}

TensorFuseStatus MemoryManager::memset_device(void* ptr, int value, size_t size, cudaStream_t stream) {
    cudaError_t error;
    
    if (stream) {
        error = cudaMemsetAsync(ptr, value, size, stream);
    } else {
        error = cudaMemset(ptr, value, size);
    }
    
    return (error == cudaSuccess) ? TENSORFUSE_SUCCESS : TENSORFUSE_ERROR_CUDA_ERROR;
}

MemoryStats MemoryManager::get_memory_stats() const {
    if (use_memory_pool_ && global_pool) {
        return global_pool->get_stats();
    } else {
        MemoryStats stats{};
        
        // Get CUDA memory info
        size_t free_bytes, total_bytes;
        cudaError_t error = cudaMemGetInfo(&free_bytes, &total_bytes);
        
        if (error == cudaSuccess) {
            stats.total_allocated = total_bytes;
            stats.free_memory = free_bytes;
            stats.used_memory = total_bytes - free_bytes;
        }
        
        return stats;
    }
}

void MemoryManager::clear_memory_pool() {
    if (global_pool) {
        global_pool->clear();
    }
}

void MemoryManager::set_memory_pool_enabled(bool enabled) {
    use_memory_pool_ = enabled;
}

bool MemoryManager::is_memory_pool_enabled() const {
    return use_memory_pool_;
}

size_t MemoryManager::align_size(size_t size, size_t alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
}

// Global memory manager instance
static MemoryManager* global_memory_manager = nullptr;
static std::once_flag memory_manager_init_flag;

void initialize_memory_manager() {
    std::call_once(memory_manager_init_flag, []() {
        global_memory_manager = new MemoryManager();
        
        // Register cleanup at exit
        std::atexit([]() {
            delete global_memory_manager;
            global_memory_manager = nullptr;
        });
    });
}

MemoryManager& get_memory_manager() {
    initialize_memory_manager();
    return *global_memory_manager;
}

} // namespace runtime
} // namespace tensorfuse 