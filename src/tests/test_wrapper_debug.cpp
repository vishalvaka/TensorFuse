#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>

// Forward declarations
typedef enum {
    TENSORFUSE_SUCCESS = 0,
    TENSORFUSE_ERROR_CUDA_ERROR,
    TENSORFUSE_ERROR_KERNEL_LAUNCH_FAILED,
    TENSORFUSE_ERROR_UNSUPPORTED_OPERATION
} TensorFuseStatus;

extern "C" {
    TensorFuseStatus fused_gemm_bias_gelu_fp32_wrapper(
        const void* A, const void* B, const void* bias, void* C,
        int M, int N, int K, float alpha, float beta, cudaStream_t stream);
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

int main() {
    std::cout << "Testing wrapper functions directly..." << std::endl;
    
    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0));
    
    // Test case 1: Simple 1x1 matrix (same as Python test)
    std::cout << "Test 1: 1x1 matrix" << std::endl;
    
    const int M = 1, N = 1, K = 1;
    const float alpha = 1.0f, beta = 1.0f;
    
    // Test data: A=[[1]], B=[[2]], bias=[0.5]
    std::vector<float> A = {1.0f};
    std::vector<float> B = {2.0f};
    std::vector<float> bias = {0.5f};
    std::vector<float> C_gpu(M * N, 0.0f);
    std::vector<float> C_ref(M * N, 0.0f);
    
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
    
    // Clear output memory
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
    
    // Launch wrapper function
    TensorFuseStatus status = fused_gemm_bias_gelu_fp32_wrapper(
        d_A, d_B, d_bias, d_C, M, N, K, alpha, beta, nullptr);
    
    if (status != TENSORFUSE_SUCCESS) {
        std::cerr << "Wrapper function failed with status: " << status << std::endl;
        return 1;
    }
    
    // Synchronize and copy result back
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compute reference
    gemm_bias_gelu_ref(A.data(), B.data(), bias.data(), C_ref.data(), M, N, K, alpha, beta);
    
    // Print results
    std::cout << "Input: A=" << A[0] << ", B=" << B[0] << ", bias=" << bias[0] << std::endl;
    std::cout << "Expected: GELU(1*2 + 0.5) = GELU(2.5) = " << C_ref[0] << std::endl;
    std::cout << "GPU result: " << C_gpu[0] << std::endl;
    
    float error = std::abs(C_gpu[0] - C_ref[0]);
    std::cout << "Error: " << error << std::endl;
    
    if (error < 1e-4f) {
        std::cout << "✅ SUCCESS!" << std::endl;
    } else {
        std::cout << "❌ FAIL!" << std::endl;
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_C));
    
    return 0;
} 