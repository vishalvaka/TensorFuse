/*
 * Copyright (c) 2024, TensorFuse Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Simple test to verify TensorFuse library can be loaded
 */

#include <iostream>
#include <dlfcn.h>
#include <cuda_runtime.h>

int main() {
    std::cout << "TensorFuse Library Test" << std::endl;
    std::cout << "======================" << std::endl;
    
    // Check CUDA device
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        std::cerr << "CUDA not available: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    // Get device properties
    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, 0) == cudaSuccess) {
        std::cout << "GPU: " << props.name << std::endl;
        std::cout << "Compute Capability: " << props.major << "." << props.minor << std::endl;
    }
    
    // Try to load the TensorFuse library from multiple possible paths
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
        std::cerr << "Cannot load TensorFuse library from any path: " << dlerror() << std::endl;
        return 1;
    }
    
    std::cout << "TensorFuse library loaded successfully!" << std::endl;
    
    // Try to find the function symbols (using mangled names)
    typedef int (*fp32_func_t)(const void*, const void*, const void*, void*, int, int, int, float, float, void*);
    typedef int (*fp16_func_t)(const void*, const void*, const void*, void*, int, int, int, float, float, void*);
    
    fp32_func_t fp32_func = (fp32_func_t)dlsym(handle, "_ZN10tensorfuse7kernels25fused_gemm_bias_gelu_fp32EPKvS2_S2_PviiiffP11CUstream_st");
    fp16_func_t fp16_func = (fp16_func_t)dlsym(handle, "_ZN10tensorfuse7kernels25fused_gemm_bias_gelu_fp16EPKvS2_S2_PviiiffP11CUstream_st");
    
    if (fp32_func) {
        std::cout << "Found FP32 fused_gemm_bias_gelu function" << std::endl;
    } else {
        std::cout << "FP32 function not found: " << dlerror() << std::endl;
    }
    
    if (fp16_func) {
        std::cout << "Found FP16 fused_gemm_bias_gelu function" << std::endl;
    } else {
        std::cout << "FP16 function not found: " << dlerror() << std::endl;
    }
    
    // Clean up
    dlclose(handle);
    
    if (fp32_func && fp16_func) {
        std::cout << std::endl << "SUCCESS: All kernel functions found in library!" << std::endl;
        return 0;
    } else {
        std::cout << std::endl << "FAILURE: Some kernel functions missing!" << std::endl;
        return 1;
    }
} 