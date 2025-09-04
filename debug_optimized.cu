#include <iostream>
#include <cuda_runtime.h>
#include "src/kernels/fused_gemm_bias_gelu_optimized.cu"

int main() {
    std::cout << "Testing optimized kernel directly..." << std::endl;
    
    // Simple test case
    const int M = 32, N = 32, K = 32;
    const float alpha = 1.0f, beta = 1.0f;
    
    // Allocate GPU memory
    float *d_A, *d_B, *d_bias, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_bias, N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // Initialize with simple values
    cudaMemset(d_A, 0, M * K * sizeof(float));
    cudaMemset(d_B, 0, K * N * sizeof(float));
    cudaMemset(d_bias, 0, N * sizeof(float));
    cudaMemset(d_C, 0, M * N * sizeof(float));
    
    // Test FP32 optimized kernel
    std::cout << "Calling optimized_fused_gemm_bias_gelu_fp32..." << std::endl;
    TensorFuseStatus status = optimized_fused_gemm_bias_gelu_fp32(
        d_A, d_B, d_bias, d_C, M, N, K, alpha, beta, nullptr);
    
    std::cout << "Status: " << status << std::endl;
    if (status == TENSORFUSE_SUCCESS) {
        std::cout << "SUCCESS!" << std::endl;
    } else {
        std::cout << "FAILED with status: " << status << std::endl;
    }
    
    // Check CUDA error
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(cuda_error) << std::endl;
    }
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_bias);
    cudaFree(d_C);
    
    return 0;
}
