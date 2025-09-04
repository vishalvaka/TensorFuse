/*
 * Copyright (c) 2024, TensorFuse Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Runtime context for managing CUDA streams, device state, and execution environment
 */

#include "runtime/runtime_context.h"
#include "runtime/memory_manager.h"
#include "utils/error_handling.h"
#include "utils/cuda_utils.h"

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <curand.h>

namespace tensorfuse {
namespace runtime {

// RuntimeContext implementation
RuntimeContext::RuntimeContext() 
    : device_id_(-1)
    , compute_capability_major_(0)
    , compute_capability_minor_(0)
    , main_stream_(nullptr)
    , is_initialized_(false) {
}

RuntimeContext::~RuntimeContext() {
    if (is_initialized_) {
        cleanup();
    }
}

TensorFuseStatus RuntimeContext::initialize(int device_id) {
    if (is_initialized_) {
        return TENSORFUSE_ERROR_INVALID_CONFIGURATION;
    }
    
    // Set CUDA device
    if (device_id >= 0) {
        cudaError_t error = cudaSetDevice(device_id);
        if (error != cudaSuccess) {
            return TENSORFUSE_ERROR_CUDA_ERROR;
        }
        device_id_ = device_id;
    } else {
        // Use current device
        cudaError_t error = cudaGetDevice(&device_id_);
        if (error != cudaSuccess) {
            return TENSORFUSE_ERROR_CUDA_ERROR;
        }
    }
    
    // Get device properties
    cudaDeviceProp props;
    cudaError_t error = cudaGetDeviceProperties(&props, device_id_);
    if (error != cudaSuccess) {
        return TENSORFUSE_ERROR_CUDA_ERROR;
    }
    
    device_properties_ = props;
    compute_capability_major_ = props.major;
    compute_capability_minor_ = props.minor;
    
    // Check if device supports required compute capability
    if (compute_capability_major_ < 8) {
        return TENSORFUSE_ERROR_UNSUPPORTED_OPERATION;
    }
    
    // Create main CUDA stream
    error = cudaStreamCreate(&main_stream_);
    if (error != cudaSuccess) {
        return TENSORFUSE_ERROR_CUDA_ERROR;
    }
    
    // Initialize cuBLAS
    cublasStatus_t cublas_status = cublasCreate(&cublas_handle_);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        return TENSORFUSE_ERROR_CUDA_ERROR;
    }
    
    cublas_status = cublasSetStream(cublas_handle_, main_stream_);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        return TENSORFUSE_ERROR_CUDA_ERROR;
    }
    
    // Initialize cuDNN
    cudnnStatus_t cudnn_status = cudnnCreate(&cudnn_handle_);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        return TENSORFUSE_ERROR_CUDA_ERROR;
    }
    
    cudnn_status = cudnnSetStream(cudnn_handle_, main_stream_);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        return TENSORFUSE_ERROR_CUDA_ERROR;
    }
    
    // Initialize cuRAND
    curandStatus_t curand_status = curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT);
    if (curand_status != CURAND_STATUS_SUCCESS) {
        return TENSORFUSE_ERROR_CUDA_ERROR;
    }
    
    curand_status = curandSetStream(curand_generator_, main_stream_);
    if (curand_status != CURAND_STATUS_SUCCESS) {
        return TENSORFUSE_ERROR_CUDA_ERROR;
    }
    
    // Set random seed
    curand_status = curandSetPseudoRandomGeneratorSeed(curand_generator_, 42);
    if (curand_status != CURAND_STATUS_SUCCESS) {
        return TENSORFUSE_ERROR_CUDA_ERROR;
    }
    
    // Initialize memory manager
    memory_manager_ = &get_memory_manager();
    
    is_initialized_ = true;
    return TENSORFUSE_SUCCESS;
}

void RuntimeContext::cleanup() {
    if (!is_initialized_) return;
    
    // Synchronize all operations
    if (main_stream_) {
        cudaStreamSynchronize(main_stream_);
    }
    
    // Cleanup cuRAND
    if (curand_generator_) {
        curandDestroyGenerator(curand_generator_);
        curand_generator_ = nullptr;
    }
    
    // Cleanup cuDNN
    if (cudnn_handle_) {
        cudnnDestroy(cudnn_handle_);
        cudnn_handle_ = nullptr;
    }
    
    // Cleanup cuBLAS
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
        cublas_handle_ = nullptr;
    }
    
    // Destroy additional streams
    for (auto stream : additional_streams_) {
        cudaStreamDestroy(stream);
    }
    additional_streams_.clear();
    
    // Destroy main stream
    if (main_stream_) {
        cudaStreamDestroy(main_stream_);
        main_stream_ = nullptr;
    }
    
    is_initialized_ = false;
}

cudaStream_t RuntimeContext::get_main_stream() const {
    return main_stream_;
}

cudaStream_t RuntimeContext::create_stream() {
    cudaStream_t stream;
    cudaError_t error = cudaStreamCreate(&stream);
    if (error == cudaSuccess) {
        additional_streams_.push_back(stream);
        return stream;
    }
    return nullptr;
}

void RuntimeContext::destroy_stream(cudaStream_t stream) {
    if (stream == main_stream_) return; // Don't destroy main stream
    
    auto it = std::find(additional_streams_.begin(), additional_streams_.end(), stream);
    if (it != additional_streams_.end()) {
        cudaStreamDestroy(stream);
        additional_streams_.erase(it);
    }
}

TensorFuseStatus RuntimeContext::synchronize() {
    cudaError_t error = cudaDeviceSynchronize();
    return (error == cudaSuccess) ? TENSORFUSE_SUCCESS : TENSORFUSE_ERROR_CUDA_ERROR;
}

TensorFuseStatus RuntimeContext::synchronize_stream(cudaStream_t stream) {
    cudaError_t error = cudaStreamSynchronize(stream ? stream : main_stream_);
    return (error == cudaSuccess) ? TENSORFUSE_SUCCESS : TENSORFUSE_ERROR_CUDA_ERROR;
}

int RuntimeContext::get_device_id() const {
    return device_id_;
}

const cudaDeviceProp& RuntimeContext::get_device_properties() const {
    return device_properties_;
}

int RuntimeContext::get_compute_capability_major() const {
    return compute_capability_major_;
}

int RuntimeContext::get_compute_capability_minor() const {
    return compute_capability_minor_;
}

bool RuntimeContext::supports_tensor_cores() const {
    return compute_capability_major_ >= 8 || 
           (compute_capability_major_ == 7 && compute_capability_minor_ >= 5);
}

bool RuntimeContext::supports_fp16() const {
    return compute_capability_major_ >= 7 || 
           (compute_capability_major_ == 6 && compute_capability_minor_ >= 1);
}

bool RuntimeContext::supports_bf16() const {
    return compute_capability_major_ >= 8;
}

bool RuntimeContext::supports_fp8() const {
    return compute_capability_major_ >= 9;
}

size_t RuntimeContext::get_max_shared_memory_per_block() const {
    return device_properties_.sharedMemPerBlock;
}

int RuntimeContext::get_max_threads_per_block() const {
    return device_properties_.maxThreadsPerBlock;
}

int RuntimeContext::get_multiprocessor_count() const {
    return device_properties_.multiProcessorCount;
}

size_t RuntimeContext::get_total_global_memory() const {
    return device_properties_.totalGlobalMem;
}

cublasHandle_t RuntimeContext::get_cublas_handle() const {
    return cublas_handle_;
}

cudnnHandle_t RuntimeContext::get_cudnn_handle() const {
    return cudnn_handle_;
}

curandGenerator_t RuntimeContext::get_curand_generator() const {
    return curand_generator_;
}

MemoryManager& RuntimeContext::get_memory_manager() const {
    return *memory_manager_;
}

DeviceInfo RuntimeContext::get_device_info() const {
    DeviceInfo info{};
    info.device_id = device_id_;
    info.device_name = device_properties_.name;
    info.compute_capability_major = compute_capability_major_;
    info.compute_capability_minor = compute_capability_minor_;
    info.total_global_memory = device_properties_.totalGlobalMem;
    info.multiprocessor_count = device_properties_.multiProcessorCount;
    info.max_threads_per_block = device_properties_.maxThreadsPerBlock;
    info.max_shared_memory_per_block = device_properties_.sharedMemPerBlock;
    info.supports_tensor_cores = supports_tensor_cores();
    info.supports_fp16 = supports_fp16();
    info.supports_bf16 = supports_bf16();
    info.supports_fp8 = supports_fp8();
    return info;
}

TensorFuseStatus RuntimeContext::set_random_seed(unsigned long long seed) {
    curandStatus_t status = curandSetPseudoRandomGeneratorSeed(curand_generator_, seed);
    return (status == CURAND_STATUS_SUCCESS) ? TENSORFUSE_SUCCESS : TENSORFUSE_ERROR_CUDA_ERROR;
}

bool RuntimeContext::is_initialized() const {
    return is_initialized_;
}

// Global runtime context management
static RuntimeContext* global_context = nullptr;
static std::once_flag context_init_flag;
static std::mutex context_mutex;

TensorFuseStatus initialize_runtime_context(int device_id) {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    if (global_context && global_context->is_initialized()) {
        return TENSORFUSE_ERROR_INVALID_CONFIGURATION;
    }
    
    std::call_once(context_init_flag, []() {
        global_context = new RuntimeContext();
        
        // Register cleanup at exit
        std::atexit([]() {
            if (global_context) {
                global_context->cleanup();
                delete global_context;
                global_context = nullptr;
            }
        });
    });
    
    return global_context->initialize(device_id);
}

RuntimeContext& get_runtime_context() {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    if (!global_context) {
        // Auto-initialize with default device
        std::call_once(context_init_flag, []() {
            global_context = new RuntimeContext();
        });
        
        if (!global_context->is_initialized()) {
            global_context->initialize(-1); // Use current device
        }
    }
    
    return *global_context;
}

void cleanup_runtime_context() {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    if (global_context) {
        global_context->cleanup();
    }
}

} // namespace runtime
} // namespace tensorfuse 