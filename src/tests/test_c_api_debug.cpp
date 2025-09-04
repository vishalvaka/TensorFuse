#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>

// Include TensorFuse headers
#include "../tensorfuse/types.h"
#include "../tensorfuse/tensorfuse.h"

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
    std::cout << "Testing C API function directly..." << std::endl;
    
    // Initialize TensorFuse
    TensorFuseConfig config = {};
    config.device_count = 1;
    config.device_ids = nullptr;
    config.workspace_size_bytes = 1024 * 1024 * 1024; // 1GB
    config.enable_profiling = false;
    config.enable_autotuning = true;
    config.enable_fp8 = false;
    config.log_level = 1;
    config.cache_dir = "./cache";
    
    TensorFuseStatus status = tensorfuse_init(&config);
    if (status != TENSORFUSE_SUCCESS) {
        std::cerr << "Failed to initialize TensorFuse: " << status << std::endl;
        return 1;
    }
    
    // Test case 1: Simple 1x1 matrix (same as Python test)
    std::cout << "Test 1: 1x1 matrix" << std::endl;
    
    const int M = 1, N = 1, K = 1;
    
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
    
    // Create TensorFuseTensor structures exactly like Python bindings
    TensorFuseTensor input_tensor = {};
    input_tensor.data = d_A;
    input_tensor.shape.ndims = 2;
    input_tensor.shape.dims[0] = M;
    input_tensor.shape.dims[1] = K;
    input_tensor.dtype = TENSORFUSE_FLOAT32;
    input_tensor.layout = TENSORFUSE_LAYOUT_ROW_MAJOR;
    input_tensor.device_id = 0;
    input_tensor.size_bytes = M * K * sizeof(float);
    input_tensor.is_contiguous = true;
    input_tensor.scale = nullptr;
    input_tensor.zero_point = nullptr;
    
    TensorFuseTensor weight_tensor = {};
    weight_tensor.data = d_B;
    weight_tensor.shape.ndims = 2;
    weight_tensor.shape.dims[0] = K;
    weight_tensor.shape.dims[1] = N;
    weight_tensor.dtype = TENSORFUSE_FLOAT32;
    weight_tensor.layout = TENSORFUSE_LAYOUT_ROW_MAJOR;
    weight_tensor.device_id = 0;
    weight_tensor.size_bytes = K * N * sizeof(float);
    weight_tensor.is_contiguous = true;
    weight_tensor.scale = nullptr;
    weight_tensor.zero_point = nullptr;
    
    TensorFuseTensor bias_tensor = {};
    bias_tensor.data = d_bias;
    bias_tensor.shape.ndims = 1;
    bias_tensor.shape.dims[0] = N;
    bias_tensor.dtype = TENSORFUSE_FLOAT32;
    bias_tensor.layout = TENSORFUSE_LAYOUT_ROW_MAJOR;
    bias_tensor.device_id = 0;
    bias_tensor.size_bytes = N * sizeof(float);
    bias_tensor.is_contiguous = true;
    bias_tensor.scale = nullptr;
    bias_tensor.zero_point = nullptr;
    
    TensorFuseTensor output_tensor = {};
    output_tensor.data = d_C;
    output_tensor.shape.ndims = 2;
    output_tensor.shape.dims[0] = M;
    output_tensor.shape.dims[1] = N;
    output_tensor.dtype = TENSORFUSE_FLOAT32;
    output_tensor.layout = TENSORFUSE_LAYOUT_ROW_MAJOR;
    output_tensor.device_id = 0;
    output_tensor.size_bytes = M * N * sizeof(float);
    output_tensor.is_contiguous = true;
    output_tensor.scale = nullptr;
    output_tensor.zero_point = nullptr;
    
    // Call C API function
    status = tensorfuse_fused_gemm_bias_gelu(
        &input_tensor, &weight_tensor, &bias_tensor, &output_tensor, nullptr);
    
    if (status != TENSORFUSE_SUCCESS) {
        std::cerr << "C API function failed with status: " << status << std::endl;
        std::cerr << "Error: " << tensorfuse_get_error_string(status) << std::endl;
        return 1;
    }
    
    // Synchronize and copy result back
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compute reference
    gemm_bias_gelu_ref(A.data(), B.data(), bias.data(), C_ref.data(), M, N, K, 1.0f, 1.0f);
    
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
    
    tensorfuse_shutdown();
    
    return 0;
} 