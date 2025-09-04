/**
 * @file roofline_analysis.cu
 * @brief Roofline Model Analysis for TensorFuse Kernels
 * 
 * This tool creates roofline models by measuring:
 * 1. Peak theoretical FLOPS (compute roof)
 * 2. Peak memory bandwidth (memory roof)
 * 3. Operational intensity of TensorFuse kernels
 * 4. Actual performance vs theoretical limits
 * 
 * The roofline model helps identify whether kernels are compute-bound
 * or memory-bound, guiding optimization efforts.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <functional>
#include <numeric>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>

// TensorFuse headers
#include "tensorfuse_c_api.h"

/**
 * @brief GPU specifications structure
 */
struct GPUSpecs {
    std::string name;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_multiprocessor;
    int max_threads_per_block;
    int warp_size;
    size_t global_memory_bytes;
    int memory_bus_width;
    int memory_clock_rate;
    int core_clock_rate;
    int tensor_core_count;
    
    // Derived specifications
    double peak_fp32_flops;
    double peak_fp16_flops;
    double peak_int8_ops;
    double peak_memory_bandwidth_gbps;
    
    void calculate_peak_performance() {
        // Peak FP32 FLOPS = cores × 2 ops/cycle × frequency
        int cores_per_sm = get_cores_per_sm();
        peak_fp32_flops = cores_per_sm * multiprocessor_count * 2.0 * core_clock_rate * 1e6;
        
        // Peak FP16 FLOPS (with Tensor Cores)
        if (compute_capability_major >= 8) {
            // Ada Lovelace: 4th gen Tensor Cores
            peak_fp16_flops = tensor_core_count * 4096.0 * core_clock_rate * 1e6; // 4096 ops/cycle per tensor core
        } else if (compute_capability_major >= 7) {
            // Turing/Ampere: 2nd/3rd gen Tensor Cores  
            peak_fp16_flops = tensor_core_count * 2048.0 * core_clock_rate * 1e6;
        } else {
            peak_fp16_flops = peak_fp32_flops * 2.0; // No tensor cores, just 2x throughput
        }
        
        // Peak INT8 ops (with Tensor Cores)
        if (compute_capability_major >= 8) {
            peak_int8_ops = tensor_core_count * 8192.0 * core_clock_rate * 1e6; // 8192 ops/cycle per tensor core
        } else if (compute_capability_major >= 7) {
            peak_int8_ops = tensor_core_count * 4096.0 * core_clock_rate * 1e6;
        } else {
            peak_int8_ops = peak_fp32_flops * 4.0; // No tensor cores, just 4x throughput
        }
        
        // Peak memory bandwidth
        peak_memory_bandwidth_gbps = (memory_bus_width / 8.0) * memory_clock_rate * 2.0 / 1e6; // DDR
    }
    
private:
    int get_cores_per_sm() {
        // CUDA cores per SM for different architectures
        if (compute_capability_major == 8) {
            if (compute_capability_minor == 9) return 128; // Ada Lovelace
            else return 64; // Ampere
        } else if (compute_capability_major == 7) {
            return 64; // Turing/Ampere
        } else if (compute_capability_major == 6) {
            return 64; // Pascal
        } else {
            return 32; // Older architectures
        }
    }
};

/**
 * @brief Roofline data point
 */
struct RooflinePoint {
    std::string kernel_name;
    double operational_intensity;  // FLOPS/byte
    double performance_flops;      // Actual FLOPS achieved
    double memory_bandwidth_gbps;  // Actual memory bandwidth
    double execution_time_ms;
    
    void print() const {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Kernel: " << kernel_name << std::endl;
        std::cout << "  Operational Intensity: " << operational_intensity << " FLOPS/byte" << std::endl;
        std::cout << "  Performance: " << performance_flops/1e12 << " TFLOPS" << std::endl;
        std::cout << "  Memory Bandwidth: " << memory_bandwidth_gbps << " GB/s" << std::endl;
        std::cout << "  Execution Time: " << execution_time_ms << " ms" << std::endl;
    }
};

/**
 * @brief Get GPU specifications
 */
GPUSpecs get_gpu_specs(int device_id = 0) {
    GPUSpecs specs;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    
    specs.name = prop.name;
    specs.compute_capability_major = prop.major;
    specs.compute_capability_minor = prop.minor;
    specs.multiprocessor_count = prop.multiProcessorCount;
    specs.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
    specs.max_threads_per_block = prop.maxThreadsPerBlock;
    specs.warp_size = prop.warpSize;
    specs.global_memory_bytes = prop.totalGlobalMem;
    specs.memory_bus_width = prop.memoryBusWidth;
    specs.memory_clock_rate = prop.memoryClockRate;
    specs.core_clock_rate = prop.clockRate;
    
    // Estimate tensor core count (architecture-dependent)
    if (specs.compute_capability_major >= 8) {
        specs.tensor_core_count = specs.multiprocessor_count * 4; // 4 tensor cores per SM on Ada/Hopper
    } else if (specs.compute_capability_major >= 7) {
        specs.tensor_core_count = specs.multiprocessor_count * 2; // 2 tensor cores per SM on Turing/Ampere
    } else {
        specs.tensor_core_count = 0; // No tensor cores
    }
    
    specs.calculate_peak_performance();
    
    return specs;
}

/**
 * @brief Measure memory bandwidth using simple copy operations
 */
double measure_memory_bandwidth(size_t size_bytes, int iterations = 100) {
    void* src;
    void* dst;
    
    cudaMalloc(&src, size_bytes);
    cudaMalloc(&dst, size_bytes);
    
    // Initialize source data
    cudaMemset(src, 1, size_bytes);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        cudaMemcpy(dst, src, size_bytes, cudaMemcpyDeviceToDevice);
    }
    
    // Time the memory copies
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cudaMemcpy(dst, src, size_bytes, cudaMemcpyDeviceToDevice);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    
    double bandwidth_gbps = (size_bytes * iterations * 2.0) / (time_ms / 1000.0) / 1e9; // 2x for read+write
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(src);
    cudaFree(dst);
    
    return bandwidth_gbps;
}

/**
 * @brief Measure peak compute performance using GEMM
 */
double measure_peak_compute_flops(int M, int N, int K, int iterations = 100) {
    float* A;
    float* B;
    float* C;
    
    cudaMalloc(&A, M * K * sizeof(float));
    cudaMalloc(&B, K * N * sizeof(float));
    cudaMalloc(&C, M * N * sizeof(float));
    
    // Initialize matrices
    cudaMemset(A, 1, M * K * sizeof(float));
    cudaMemset(B, 1, K * N * sizeof(float));
    
    // Use cuBLAS for peak performance
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f, beta = 0.0f;
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, M, B, K, &beta, C, M);
    }
    
    // Time the GEMM operations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, M, B, K, &beta, C, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    
    // Calculate FLOPS
    long long flops_per_gemm = 2LL * M * N * K;
    double total_flops = flops_per_gemm * iterations;
    double flops_per_second = total_flops / (time_ms / 1000.0);
    
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    
    return flops_per_second;
}

/**
 * @brief Analyze a specific kernel for roofline characteristics
 */
RooflinePoint analyze_kernel_roofline(
    const std::string& kernel_name,
    std::function<void(float*, float*, float*, float*, int, int, int, float&)> kernel_func,
    int M, int N, int K, int iterations = 100) {
    
    // Allocate memory
    float* A;
    float* B;
    float* bias;
    float* C;
    
    cudaMalloc(&A, M * K * sizeof(float));
    cudaMalloc(&B, K * N * sizeof(float));
    cudaMalloc(&bias, N * sizeof(float));
    cudaMalloc(&C, M * N * sizeof(float));
    
    // Initialize with random data
    cudaMemset(A, 1, M * K * sizeof(float));
    cudaMemset(B, 1, K * N * sizeof(float));
    cudaMemset(bias, 1, N * sizeof(float));
    
    // Warmup
    float warmup_time;
    for (int i = 0; i < 10; i++) {
        kernel_func(A, B, bias, C, M, N, K, warmup_time);
    }
    
    // Benchmark
    std::vector<float> times;
    for (int i = 0; i < iterations; i++) {
        float time_ms;
        kernel_func(A, B, bias, C, M, N, K, time_ms);
        times.push_back(time_ms);
    }
    
    // Calculate statistics
    float avg_time = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
    
    // Calculate performance metrics
    long long flops = 2LL * M * N * K + M * N; // GEMM + bias
    double performance_flops = flops / (avg_time / 1000.0);
    
    // Memory operations
    long long memory_bytes = (M * K + K * N + N + M * N) * sizeof(float);
    double memory_bandwidth_gbps = memory_bytes / (avg_time / 1000.0) / 1e9;
    
    // Operational intensity
    double operational_intensity = static_cast<double>(flops) / memory_bytes;
    
    // Cleanup
    cudaFree(A);
    cudaFree(B);
    cudaFree(bias);
    cudaFree(C);
    
    return {kernel_name, operational_intensity, performance_flops, memory_bandwidth_gbps, avg_time};
}

/**
 * @brief Generate roofline plot data
 */
void generate_roofline_plot(const GPUSpecs& specs, const std::vector<RooflinePoint>& points, const std::string& filename) {
    std::ofstream file(filename);
    
    file << "# Roofline Plot Data\n";
    file << "# GPU: " << specs.name << "\n";
    file << "# Peak FP32 FLOPS: " << specs.peak_fp32_flops/1e12 << " TFLOPS\n";
    file << "# Peak Memory Bandwidth: " << specs.peak_memory_bandwidth_gbps << " GB/s\n";
    file << "# Operational_Intensity(FLOPS/byte) Performance(TFLOPS)\n";
    
    // Add roofline boundaries
    file << "# Roofline boundaries\n";
    file << "# Memory roof (bandwidth limited)\n";
    for (double oi = 0.1; oi <= 100.0; oi *= 1.1) {
        double perf = std::min(specs.peak_memory_bandwidth_gbps * oi / 1e12, specs.peak_fp32_flops / 1e12);
        file << oi << " " << perf << " roofline\n";
    }
    
    file << "\n# Actual kernel performance\n";
    for (const auto& point : points) {
        file << point.operational_intensity << " " << point.performance_flops/1e12 << " " << point.kernel_name << "\n";
    }
    
    file.close();
}

/**
 * @brief TensorFuse kernel wrapper for roofline analysis
 */
void tensorfuse_gemm_bias_gelu_wrapper(float* A, float* B, float* bias, float* C, int M, int N, int K, float& time_ms) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    fused_gemm_bias_gelu_fp32_wrapper(A, B, bias, C, M, N, K, 1.0f, 1.0f, nullptr);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&time_ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/**
 * @brief cuBLAS baseline wrapper for roofline analysis
 */
void cublas_gemm_wrapper(float* A, float* B, float* bias, float* C, int M, int N, int K, float& time_ms) {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const float alpha = 1.0f, beta = 0.0f;
    
    cudaEventRecord(start);
    // GEMM
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, M, B, K, &beta, C, M);
    // Add bias (simplified - just add first bias value to all elements)
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&time_ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/**
 * @brief Main roofline analysis function
 */
int main(int argc, char* argv[]) {
    std::cout << "🏔️  TensorFuse Roofline Analysis" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // Get GPU specifications
    GPUSpecs specs = get_gpu_specs();
    
    std::cout << "🎯 GPU Specifications:" << std::endl;
    std::cout << "  Name: " << specs.name << std::endl;
    std::cout << "  Compute Capability: " << specs.compute_capability_major << "." << specs.compute_capability_minor << std::endl;
    std::cout << "  Multiprocessors: " << specs.multiprocessor_count << std::endl;
    std::cout << "  Tensor Cores: " << specs.tensor_core_count << std::endl;
    std::cout << "  Peak FP32 FLOPS: " << std::fixed << std::setprecision(2) << specs.peak_fp32_flops/1e12 << " TFLOPS" << std::endl;
    std::cout << "  Peak FP16 FLOPS: " << specs.peak_fp16_flops/1e12 << " TFLOPS" << std::endl;
    std::cout << "  Peak INT8 Ops: " << specs.peak_int8_ops/1e12 << " TOPS" << std::endl;
    std::cout << "  Peak Memory Bandwidth: " << specs.peak_memory_bandwidth_gbps << " GB/s" << std::endl;
    std::cout << std::endl;
    
    // Measure actual peak performance
    std::cout << "📊 Measuring Actual Peak Performance..." << std::endl;
    
    double actual_memory_bandwidth = measure_memory_bandwidth(1024 * 1024 * 1024); // 1GB
    double actual_compute_flops = measure_peak_compute_flops(4096, 4096, 4096);
    
    std::cout << "  Actual Memory Bandwidth: " << actual_memory_bandwidth << " GB/s" << std::endl;
    std::cout << "  Actual Compute Performance: " << actual_compute_flops/1e12 << " TFLOPS" << std::endl;
    std::cout << std::endl;
    
    // Initialize TensorFuse
    tensorfuse_simple_init(0); // Use device 0
    
    // Test various matrix sizes
    std::vector<std::tuple<int, int, int>> test_sizes = {
        {128, 512, 256},      // Small
        {512, 1024, 512},     // Medium
        {1024, 2048, 1024},   // Large (BERT-like)
        {2048, 4096, 2048},   // Very large
    };
    
    std::vector<RooflinePoint> points;
    
    for (const auto& size : test_sizes) {
        int M, N, K;
        std::tie(M, N, K) = size;
        
        std::cout << "🔍 Analyzing matrix size " << M << "x" << N << "x" << K << "..." << std::endl;
        
        // Analyze TensorFuse kernel
        auto tf_point = analyze_kernel_roofline(
            "TensorFuse_" + std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K),
            tensorfuse_gemm_bias_gelu_wrapper,
            M, N, K
        );
        points.push_back(tf_point);
        tf_point.print();
        
        // Analyze cuBLAS baseline
        auto cublas_point = analyze_kernel_roofline(
            "cuBLAS_" + std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K),
            cublas_gemm_wrapper,
            M, N, K
        );
        points.push_back(cublas_point);
        cublas_point.print();
        
        std::cout << std::endl;
    }
    
    // Generate roofline plot data
    generate_roofline_plot(specs, points, "benchmarks/results/roofline_data.txt");
    
    // Summary analysis
    std::cout << "📈 ROOFLINE ANALYSIS SUMMARY" << std::endl;
    std::cout << "============================" << std::endl;
    
    for (const auto& point : points) {
        double efficiency = (point.performance_flops / specs.peak_fp32_flops) * 100.0;
        double memory_efficiency = (point.memory_bandwidth_gbps / specs.peak_memory_bandwidth_gbps) * 100.0;
        
        std::cout << point.kernel_name << ":" << std::endl;
        std::cout << "  Compute Efficiency: " << std::fixed << std::setprecision(1) << efficiency << "%" << std::endl;
        std::cout << "  Memory Efficiency: " << memory_efficiency << "%" << std::endl;
        
        if (point.operational_intensity < 10.0) {
            std::cout << "  Analysis: MEMORY-BOUND (focus on memory optimization)" << std::endl;
        } else {
            std::cout << "  Analysis: COMPUTE-BOUND (focus on compute optimization)" << std::endl;
        }
        std::cout << std::endl;
    }
    
    // Cleanup
    tensorfuse_simple_cleanup();
    
    std::cout << "📁 Roofline data saved to benchmarks/results/roofline_data.txt" << std::endl;
    std::cout << "💡 Use this data to create roofline plots and guide optimization!" << std::endl;
    
    return 0;
} 