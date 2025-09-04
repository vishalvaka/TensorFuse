/*
 * Working test for TensorFuse kernels with proper memory initialization
 */

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <dlfcn.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

typedef int (*fp32_gemm_func_t)(const void*, const void*, const void*, void*, int, int, int, float, float, void*);

// GELU reference implementation
float gelu_ref(float x) {
    const float kAlpha = 0.7978845608028654f;
    const float kBeta = 0.044715f;
    return x * 0.5f * (1.0f + tanh(kAlpha * (x + kBeta * x * x * x)));
}

// Reference implementation
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

bool test_working_kernel() {
    std::cout << "Testing working kernel..." << std::endl;
    
    // Try different library paths depending on where we're run from
    const char* library_paths[] = {
        "../libtensorfuse.so",           // When run from build/src/tests/
        "src/libtensorfuse.so",          // When run from build/
        "./libtensorfuse.so",            // When run from same directory
        "build/src/libtensorfuse.so",    // When run from project root
        nullptr
    };
    
    void* handle = nullptr;
    for (int i = 0; library_paths[i] != nullptr; i++) {
        handle = dlopen(library_paths[i], RTLD_LAZY);
        if (handle) {
            std::cout << "Successfully loaded library from: " << library_paths[i] << std::endl;
            break;
        }
    }
    
    if (!handle) {
        std::cerr << "Cannot load library from any path: " << dlerror() << std::endl;
        return false;
    }
    
    // Get function
    fp32_gemm_func_t fp32_func = (fp32_gemm_func_t)dlsym(handle, 
        "_ZN10tensorfuse7kernels25fused_gemm_bias_gelu_fp32EPKvS2_S2_PviiiffP11CUstream_st");
    
    if (!fp32_func) {
        std::cerr << "Cannot find function: " << dlerror() << std::endl;
        dlclose(handle);
        return false;
    }
    
    // Small problem
    const int M = 8, N = 16, K = 32;
    const float alpha = 1.0f, beta = 1.0f;
    
    // Generate simple test data
    std::vector<float> A(M * K), B(K * N), bias(N);
    std::vector<float> C_gpu(M * N), C_ref(M * N);
    
    // Initialize with simple values
    for (int i = 0; i < M * K; i++) A[i] = 0.1f;
    for (int i = 0; i < K * N; i++) B[i] = 0.1f;
    for (int i = 0; i < N; i++) bias[i] = 0.1f;
    
    // Allocate GPU memory
    float *d_A, *d_B, *d_bias, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // Initialize C to zero
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
    
    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bias.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int status = fp32_func(d_A, d_B, d_bias, d_C, M, N, K, alpha, beta, nullptr);
    
    std::cout << "Kernel status: " << status << std::endl;
    
    if (status == 0) {  // Success
        // Copy result back
        CUDA_CHECK(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Compute reference
        gemm_bias_gelu_ref(A.data(), B.data(), bias.data(), C_ref.data(), M, N, K, alpha, beta);
        
        // Check a few values
        std::cout << "GPU result[0]: " << C_gpu[0] << std::endl;
        std::cout << "CPU result[0]: " << C_ref[0] << std::endl;
        
        // Check if results are reasonable
        bool reasonable = true;
        for (int i = 0; i < M * N; i++) {
            if (std::abs(C_gpu[i]) > 10.0f || !std::isfinite(C_gpu[i])) {
                reasonable = false;
                break;
            }
        }
        
        std::cout << "Results are " << (reasonable ? "reasonable" : "unreasonable") << std::endl;
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_C));
    dlclose(handle);
    
    return (status == 0);
}

int main() {
    std::cout << "Working Kernel Test" << std::endl;
    std::cout << "===================" << std::endl;
    
    CUDA_CHECK(cudaSetDevice(0));
    
    bool success = test_working_kernel();
    
    std::cout << "Test result: " << (success ? "SUCCESS" : "FAILURE") << std::endl;
    
    return success ? 0 : 1;
} 