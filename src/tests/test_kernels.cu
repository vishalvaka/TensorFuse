/*
 * Copyright (c) 2024, TensorFuse Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Simple test program to verify TensorFuse kernels work correctly
 */

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Include TensorFuse headers
typedef enum {
    TENSORFUSE_SUCCESS = 0,
    TENSORFUSE_ERROR_CUDA_ERROR,
    TENSORFUSE_ERROR_KERNEL_LAUNCH_FAILED,
    TENSORFUSE_ERROR_UNSUPPORTED_OPERATION
} TensorFuseStatus;

// Use the correct C++ namespace declarations
namespace tensorfuse {
namespace kernels {
    TensorFuseStatus fused_gemm_bias_gelu_fp32(
        const void* A, const void* B, const void* bias, void* C,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream);
        
    TensorFuseStatus fused_gemm_bias_gelu_fp16(
        const void* A, const void* B, const void* bias, void* C,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream);
}
}

// GELU reference implementation for validation
float gelu_ref(float x) {
    const float kAlpha = 0.7978845608028654f;  // sqrt(2/pi)
    const float kBeta = 0.044715f;
    return x * 0.5f * (1.0f + tanh(kAlpha * (x + kBeta * x * x * x)));
}

// Matrix multiplication reference implementation
void gemm_bias_gelu_ref(
    const float* A, const float* B, const float* bias, float* C,
    int M, int N, int K, float alpha, float beta) {
    
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            float result = alpha * sum + beta * bias[n];
            C[m * N + n] = gelu_ref(result);
        }
    }
}

// Check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Test FP32 kernel
bool test_fp32_kernel() {
    std::cout << "Testing FP32 kernel..." << std::endl;
    
    // Problem size
    const int M = 64, N = 128, K = 256;
    const float alpha = 1.0f, beta = 1.0f;
    
    // Initialize random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> A(M * K), B(K * N), bias(N);
    std::vector<float> C_gpu(M * N), C_ref(M * N);
    
    for (int i = 0; i < M * K; i++) A[i] = dist(gen);
    for (int i = 0; i < K * N; i++) B[i] = dist(gen);
    for (int i = 0; i < N; i++) bias[i] = dist(gen);
    
    // Allocate GPU memory
    float *d_A, *d_B, *d_bias, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bias.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    TensorFuseStatus status = tensorfuse::kernels::fused_gemm_bias_gelu_fp32(
        d_A, d_B, d_bias, d_C, M, N, K, alpha, beta, nullptr);
    
    if (status != TENSORFUSE_SUCCESS) {
        std::cerr << "FP32 kernel launch failed with status: " << status << std::endl;
        return false;
    }
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Compute reference
    gemm_bias_gelu_ref(A.data(), B.data(), bias.data(), C_ref.data(), M, N, K, alpha, beta);
    
    // Validate results
    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float error = std::abs(C_gpu[i] - C_ref[i]);
        max_error = std::max(max_error, error);
    }
    
    std::cout << "FP32 Max error: " << max_error << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_C));
    
    return max_error < 1e-4f;  // Reasonable tolerance for FP32
}

// Test FP16 kernel (simplified - just check it runs without errors)
bool test_fp16_kernel() {
    std::cout << "Testing FP16 kernel..." << std::endl;
    
    // Problem size
    const int M = 64, N = 128, K = 256;
    const float alpha = 1.0f, beta = 1.0f;
    
    // Initialize random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<__half> A(M * K), B(K * N), bias(N), C(M * N);
    
    for (int i = 0; i < M * K; i++) A[i] = __float2half(dist(gen));
    for (int i = 0; i < K * N; i++) B[i] = __float2half(dist(gen));
    for (int i = 0; i < N; i++) bias[i] = __float2half(dist(gen));
    
    // Allocate GPU memory
    __half *d_A, *d_B, *d_bias, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_bias, N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(__half)));
    
    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bias.data(), N * sizeof(__half), cudaMemcpyHostToDevice));
    
    // Launch kernel
    TensorFuseStatus status = tensorfuse::kernels::fused_gemm_bias_gelu_fp16(
        d_A, d_B, d_bias, d_C, M, N, K, alpha, beta, nullptr);
    
    if (status != TENSORFUSE_SUCCESS) {
        std::cerr << "FP16 kernel launch failed with status: " << status << std::endl;
        return false;
    }
    
    // Copy result back to verify no crashes
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, M * N * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::cout << "FP16 kernel executed successfully" << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_C));
    
    return true;
}

int main() {
    std::cout << "TensorFuse Kernel Test" << std::endl;
    std::cout << "======================" << std::endl;
    
    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0));
    
    // Get device properties
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    std::cout << "GPU: " << props.name << std::endl;
    std::cout << "Compute Capability: " << props.major << "." << props.minor << std::endl;
    std::cout << std::endl;
    
    bool all_passed = true;
    
    // Test FP32 kernel
    if (!test_fp32_kernel()) {
        std::cerr << "FP32 test FAILED" << std::endl;
        all_passed = false;
    } else {
        std::cout << "FP32 test PASSED" << std::endl;
    }
    std::cout << std::endl;
    
    // Test FP16 kernel (if supported)
    if (props.major >= 8 || (props.major == 7 && props.minor >= 5)) {
        if (!test_fp16_kernel()) {
            std::cerr << "FP16 test FAILED" << std::endl;
            all_passed = false;
        } else {
            std::cout << "FP16 test PASSED" << std::endl;
        }
    } else {
        std::cout << "FP16 tests skipped (requires compute capability 7.5+)" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "Overall result: " << (all_passed ? "PASSED" : "FAILED") << std::endl;
    
    return all_passed ? 0 : 1;
} 