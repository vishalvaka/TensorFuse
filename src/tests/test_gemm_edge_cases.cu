/*
 * Copyright (c) 2024, TensorFuse Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Edge case and stress tests for GEMM+Bias+GELU kernel
 */

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Forward declarations for TensorFuse kernels
typedef enum {
    TENSORFUSE_SUCCESS = 0,
    TENSORFUSE_ERROR_CUDA_ERROR,
    TENSORFUSE_ERROR_KERNEL_LAUNCH_FAILED,
    TENSORFUSE_ERROR_UNSUPPORTED_OPERATION,
    TENSORFUSE_ERROR_OUT_OF_MEMORY
} TensorFuseStatus;

// Use the correct C++ namespace declarations
namespace tensorfuse {
namespace kernels {
    TensorFuseStatus fused_gemm_bias_gelu_fp32(
        const void* A, const void* B, const void* bias, void* C,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream);
}
}

// GELU reference implementation
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

// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Test with different matrix sizes
bool test_different_sizes() {
    std::cout << "Testing different matrix sizes..." << std::endl;
    
    // Test cases: {M, N, K}
    std::vector<std::tuple<int, int, int>> sizes = {
        {1, 1, 1},          // Minimal case
        {1, 16, 1},         // Vector-like
        {16, 1, 16},        // Column vector
        {32, 32, 32},       // Small square
        {64, 128, 256},     // Standard case
        {128, 512, 1024},   // Medium case
        {256, 1024, 2048}   // Large case
    };
    
    for (auto& size : sizes) {
        int M = std::get<0>(size);
        int N = std::get<1>(size);
        int K = std::get<2>(size);
        
        std::cout << "  Testing M=" << M << ", N=" << N << ", K=" << K << std::endl;
        
        const float alpha = 1.0f, beta = 1.0f;
        
        // Initialize test data
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        
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
            std::cerr << "    FAILED: Kernel launch failed with status: " << status << std::endl;
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_bias));
            CUDA_CHECK(cudaFree(d_C));
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
        
        std::cout << "    Max error: " << max_error << std::endl;
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_bias));
        CUDA_CHECK(cudaFree(d_C));
        
        if (max_error > 1e-4f) {
            std::cerr << "    FAILED: Error too large!" << std::endl;
            return false;
        }
        
        std::cout << "    PASSED" << std::endl;
    }
    
    return true;
}

// Test with different alpha/beta values
bool test_alpha_beta_values() {
    std::cout << "Testing different alpha/beta values..." << std::endl;
    
    // Test cases: {alpha, beta}
    std::vector<std::pair<float, float>> alpha_beta_values = {
        {0.0f, 0.0f},     // Zero scaling
        {0.0f, 1.0f},     // Only bias
        {1.0f, 0.0f},     // Only GEMM
        {1.0f, 1.0f},     // Standard case
        {2.0f, 0.5f},     // Non-unit scaling
        {-1.0f, 1.0f},    // Negative alpha
        {1.0f, -1.0f},    // Negative beta
        {0.1f, 10.0f},    // Small alpha, large beta
        {10.0f, 0.1f}     // Large alpha, small beta
    };
    
    const int M = 32, N = 64, K = 128;
    
    for (auto& ab : alpha_beta_values) {
        float alpha = ab.first;
        float beta = ab.second;
        
        std::cout << "  Testing alpha=" << alpha << ", beta=" << beta << std::endl;
        
        // Initialize test data
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        
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
            std::cerr << "    FAILED: Kernel launch failed with status: " << status << std::endl;
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_bias));
            CUDA_CHECK(cudaFree(d_C));
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
        
        std::cout << "    Max error: " << max_error << std::endl;
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_bias));
        CUDA_CHECK(cudaFree(d_C));
        
        if (max_error > 1e-4f) {
            std::cerr << "    FAILED: Error too large!" << std::endl;
            return false;
        }
        
        std::cout << "    PASSED" << std::endl;
    }
    
    return true;
}

// Test with extreme input values
bool test_extreme_values() {
    std::cout << "Testing extreme input values..." << std::endl;
    
    const int M = 16, N = 32, K = 64;
    const float alpha = 1.0f, beta = 1.0f;
    
    // Test cases with extreme values
    std::vector<std::string> test_names = {
        "All zeros",
        "All ones", 
        "Large positive values",
        "Large negative values",
        "Very small values",
        "Mixed extreme values"
    };
    
    for (size_t test_idx = 0; test_idx < test_names.size(); test_idx++) {
        std::cout << "  Testing: " << test_names[test_idx] << std::endl;
        
        std::vector<float> A(M * K), B(K * N), bias(N);
        std::vector<float> C_gpu(M * N), C_ref(M * N);
        
        // Initialize based on test case
        switch (test_idx) {
            case 0: // All zeros
                std::fill(A.begin(), A.end(), 0.0f);
                std::fill(B.begin(), B.end(), 0.0f);
                std::fill(bias.begin(), bias.end(), 0.0f);
                break;
            case 1: // All ones
                std::fill(A.begin(), A.end(), 1.0f);
                std::fill(B.begin(), B.end(), 1.0f);
                std::fill(bias.begin(), bias.end(), 1.0f);
                break;
            case 2: // Large positive values
                std::fill(A.begin(), A.end(), 100.0f);
                std::fill(B.begin(), B.end(), 100.0f);
                std::fill(bias.begin(), bias.end(), 100.0f);
                break;
            case 3: // Large negative values
                std::fill(A.begin(), A.end(), -100.0f);
                std::fill(B.begin(), B.end(), -100.0f);
                std::fill(bias.begin(), bias.end(), -100.0f);
                break;
            case 4: // Very small values
                std::fill(A.begin(), A.end(), 1e-6f);
                std::fill(B.begin(), B.end(), 1e-6f);
                std::fill(bias.begin(), bias.end(), 1e-6f);
                break;
            case 5: // Mixed extreme values
                for (int i = 0; i < M * K; i++) A[i] = (i % 2 == 0) ? 1000.0f : -1000.0f;
                for (int i = 0; i < K * N; i++) B[i] = (i % 3 == 0) ? 1000.0f : -1000.0f;
                for (int i = 0; i < N; i++) bias[i] = (i % 2 == 0) ? 100.0f : -100.0f;
                break;
        }
        
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
            std::cerr << "    FAILED: Kernel launch failed with status: " << status << std::endl;
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_bias));
            CUDA_CHECK(cudaFree(d_C));
            return false;
        }
        
        // Copy result back
        CUDA_CHECK(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Compute reference
        gemm_bias_gelu_ref(A.data(), B.data(), bias.data(), C_ref.data(), M, N, K, alpha, beta);
        
        // Check for NaN/Inf
        bool has_invalid = false;
        for (int i = 0; i < M * N; i++) {
            if (!std::isfinite(C_gpu[i])) {
                has_invalid = true;
                break;
            }
        }
        
        if (has_invalid) {
            std::cerr << "    FAILED: Result contains NaN/Inf!" << std::endl;
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_bias));
            CUDA_CHECK(cudaFree(d_C));
            return false;
        }
        
        // Validate results (with relaxed tolerance for extreme values)
        float max_error = 0.0f;
        for (int i = 0; i < M * N; i++) {
            float error = std::abs(C_gpu[i] - C_ref[i]);
            max_error = std::max(max_error, error);
        }
        
        std::cout << "    Max error: " << max_error << std::endl;
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_bias));
        CUDA_CHECK(cudaFree(d_C));
        
        // Use more relaxed tolerance for extreme values
        float tolerance = (test_idx >= 2) ? 1e-2f : 1e-4f;
        if (max_error > tolerance) {
            std::cerr << "    FAILED: Error too large!" << std::endl;
            return false;
        }
        
        std::cout << "    PASSED" << std::endl;
    }
    
    return true;
}

int main() {
    std::cout << "TensorFuse GEMM+Bias+GELU Edge Case Tests" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0));
    
    // Get device properties
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    std::cout << "GPU: " << props.name << std::endl;
    std::cout << "Compute Capability: " << props.major << "." << props.minor << std::endl;
    std::cout << std::endl;
    
    bool all_passed = true;
    
    // Run all tests
    if (!test_different_sizes()) {
        all_passed = false;
    }
    std::cout << std::endl;
    
    if (!test_alpha_beta_values()) {
        all_passed = false;
    }
    std::cout << std::endl;
    
    if (!test_extreme_values()) {
        all_passed = false;
    }
    std::cout << std::endl;
    
    std::cout << "Overall result: " << (all_passed ? "PASSED" : "FAILED") << std::endl;
    
    return all_passed ? 0 : 1;
} 