/**
 * @file tensorfuse_wrappers.cpp
 * @brief Simple wrapper functions for TensorFuse benchmarks
 * 
 * This file provides simple wrappers that avoid header conflicts
 * by directly declaring the functions we need.
 */

#include "tensorfuse_c_api.h"
#include <cuda_runtime.h>
#include <iostream>

// Forward declarations of the actual wrapper functions from TensorFuse
extern "C" {
    // These functions are actually implemented in the TensorFuse library
    TensorFuseStatus fused_gemm_bias_gelu_fp32_wrapper(
        const void* A, const void* B, const void* bias, void* C,
        int M, int N, int K, float alpha, float beta, cudaStream_t stream);
    
    TensorFuseStatus fused_gemm_bias_gelu_fp16_wrapper(
        const void* A, const void* B, const void* bias, void* C,
        int M, int N, int K, float alpha, float beta, cudaStream_t stream);
    
    TensorFuseStatus fused_gemm_bias_gelu_int8_wrapper(
        const void* A, const void* B, const void* bias, void* C,
        int M, int N, int K, float alpha, float beta, cudaStream_t stream);
}

// Simple initialization wrapper
TensorFuseStatus tensorfuse_simple_init(int device_id) {
    // Initialize CUDA
    cudaError_t cuda_error = cudaSetDevice(device_id);
    if (cuda_error != cudaSuccess) {
        std::cerr << "CUDA setDevice failed: " << cudaGetErrorString(cuda_error) << std::endl;
        return TENSORFUSE_CUDA_ERROR;
    }
    
    std::cout << "TensorFuse initialized on device " << device_id << std::endl;
    return TENSORFUSE_SUCCESS;
}

// Simple cleanup wrapper
void tensorfuse_simple_cleanup(void) {
    std::cout << "TensorFuse cleanup completed" << std::endl;
    // Don't call cudaDeviceReset as it might interfere with other operations
} 