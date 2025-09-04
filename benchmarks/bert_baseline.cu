/**
 * @file bert_baseline.cu
 * @brief BERT-base Baseline Benchmark
 * 
 * This benchmark establishes the performance baseline for BERT-base inference
 * by comparing TensorFuse fused operations against cuBLASLt + cuDNN reference
 * implementations.
 * 
 * BERT-base Configuration:
 * - Batch size: 32
 * - Sequence length: 128  
 * - Hidden dimension: 768
 * - FFN dimension: 3072
 * - Attention heads: 12
 * - Head dimension: 64
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <cmath>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cudnn.h>

// TensorFuse headers
#include "tensorfuse_c_api.h"

// BERT-base dimensions
constexpr int BATCH_SIZE = 32;
constexpr int SEQ_LEN = 128;
constexpr int HIDDEN_DIM = 768;
constexpr int FFN_DIM = 3072;
// constexpr int NUM_HEADS = 12;  // Currently unused
// constexpr int HEAD_DIM = 64;   // Currently unused

// Benchmark configuration
constexpr int WARMUP_ITERATIONS = 10;
constexpr int BENCHMARK_ITERATIONS = 100;

/**
 * @brief CUDA kernels for bias addition and GELU
 */
__global__ void add_bias_gelu_kernel(float* output, const float* bias, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int col = idx % N;
        float x = output[idx] + bias[col];
        
        // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float kAlpha = 0.7978845608028654f;  // sqrt(2/pi)
        const float kBeta = 0.044715f;
        output[idx] = x * 0.5f * (1.0f + tanh(kAlpha * (x + kBeta * x * x * x)));
    }
}

__global__ void add_bias_kernel(float* output, const float* bias, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int col = idx % N;
        output[idx] += bias[col];
    }
}

/**
 * @brief BERT-base layer operations for benchmarking
 */
struct BertLayer {
    // Attention weights
    float* q_weight;  // [hidden_dim, hidden_dim]
    float* k_weight;  // [hidden_dim, hidden_dim]
    float* v_weight;  // [hidden_dim, hidden_dim]
    float* o_weight;  // [hidden_dim, hidden_dim]
    
    // FFN weights
    float* ffn_up_weight;    // [hidden_dim, ffn_dim]
    float* ffn_down_weight;  // [ffn_dim, hidden_dim]
    
    // Biases
    float* q_bias;     // [hidden_dim]
    float* k_bias;     // [hidden_dim]
    float* v_bias;     // [hidden_dim]
    float* o_bias;     // [hidden_dim]
    float* ffn_up_bias;   // [ffn_dim]
    float* ffn_down_bias; // [hidden_dim]
    
    BertLayer() {
        // Allocate GPU memory for weights and biases
        cudaMalloc(&q_weight, HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&k_weight, HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&v_weight, HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&o_weight, HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&ffn_up_weight, HIDDEN_DIM * FFN_DIM * sizeof(float));
        cudaMalloc(&ffn_down_weight, FFN_DIM * HIDDEN_DIM * sizeof(float));
        
        cudaMalloc(&q_bias, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&k_bias, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&v_bias, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&o_bias, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&ffn_up_bias, FFN_DIM * sizeof(float));
        cudaMalloc(&ffn_down_bias, HIDDEN_DIM * sizeof(float));
        
        // Initialize with random values
        initialize_weights();
    }
    
    ~BertLayer() {
        cudaFree(q_weight);
        cudaFree(k_weight);
        cudaFree(v_weight);
        cudaFree(o_weight);
        cudaFree(ffn_up_weight);
        cudaFree(ffn_down_weight);
        cudaFree(q_bias);
        cudaFree(k_bias);
        cudaFree(v_bias);
        cudaFree(o_bias);
        cudaFree(ffn_up_bias);
        cudaFree(ffn_down_bias);
    }
    
private:
    void initialize_weights() {
        // Initialize weights with Xavier/Glorot initialization
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // Initialize attention weights
        initialize_tensor(q_weight, HIDDEN_DIM, HIDDEN_DIM, gen);
        initialize_tensor(k_weight, HIDDEN_DIM, HIDDEN_DIM, gen);
        initialize_tensor(v_weight, HIDDEN_DIM, HIDDEN_DIM, gen);
        initialize_tensor(o_weight, HIDDEN_DIM, HIDDEN_DIM, gen);
        
        // Initialize FFN weights
        initialize_tensor(ffn_up_weight, HIDDEN_DIM, FFN_DIM, gen);
        initialize_tensor(ffn_down_weight, FFN_DIM, HIDDEN_DIM, gen);
        
        // Initialize biases (small values)
        initialize_bias(q_bias, HIDDEN_DIM, gen);
        initialize_bias(k_bias, HIDDEN_DIM, gen);
        initialize_bias(v_bias, HIDDEN_DIM, gen);
        initialize_bias(o_bias, HIDDEN_DIM, gen);
        initialize_bias(ffn_up_bias, FFN_DIM, gen);
        initialize_bias(ffn_down_bias, HIDDEN_DIM, gen);
    }
    
    void initialize_tensor(float* tensor, int dim1, int dim2, std::mt19937& gen) {
        float stddev = std::sqrt(2.0f / (dim1 + dim2));
        std::normal_distribution<float> dist(0.0f, stddev);
        
        std::vector<float> host_data(dim1 * dim2);
        for (int i = 0; i < dim1 * dim2; i++) {
            host_data[i] = dist(gen);
        }
        
        cudaMemcpy(tensor, host_data.data(), dim1 * dim2 * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    void initialize_bias(float* bias, int dim, std::mt19937& gen) {
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        
        std::vector<float> host_data(dim);
        for (int i = 0; i < dim; i++) {
            host_data[i] = dist(gen);
        }
        
        cudaMemcpy(bias, host_data.data(), dim * sizeof(float), cudaMemcpyHostToDevice);
    }
};

/**
 * @brief cuBLASLt baseline implementation
 */
class CuBLASLtBaseline {
private:
    cublasLtHandle_t handle;
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t A_desc, B_desc, C_desc;
    
public:
    CuBLASLtBaseline() {
        cublasLtCreate(&handle);
        cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    }
    
    ~CuBLASLtBaseline() {
        cublasLtMatmulDescDestroy(matmul_desc);
        cublasLtDestroy(handle);
    }
    
    float benchmark_ffn_layer(const BertLayer& layer, float* input, float* output, cudaStream_t stream) {
        // Temporary buffers
        float* intermediate;
        cudaMalloc(&intermediate, BATCH_SIZE * SEQ_LEN * FFN_DIM * sizeof(float));
        
        // Time the operations
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start, stream);
        
        // FFN Up projection: input @ ffn_up_weight + ffn_up_bias
        gemm_bias_gelu_baseline(
            input, layer.ffn_up_weight, layer.ffn_up_bias, intermediate,
            BATCH_SIZE * SEQ_LEN, FFN_DIM, HIDDEN_DIM, stream
        );
        
        // FFN Down projection: intermediate @ ffn_down_weight + ffn_down_bias
        gemm_bias_baseline(
            intermediate, layer.ffn_down_weight, layer.ffn_down_bias, output,
            BATCH_SIZE * SEQ_LEN, HIDDEN_DIM, FFN_DIM, stream
        );
        
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        
        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(intermediate);
        
        return time_ms;
    }
    
private:
    void gemm_bias_gelu_baseline(
        const float* A, const float* B, const float* bias, float* C,
        int M, int N, int K, cudaStream_t stream) {
        
        // Setup matrix layouts
        cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_32F, M, K, M);
        cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_32F, K, N, K);
        cublasLtMatrixLayoutCreate(&C_desc, CUDA_R_32F, M, N, M);
        
        // GEMM: C = A @ B
        const float alpha = 1.0f, beta = 0.0f;
        cublasLtMatmul(
            handle, matmul_desc,
            &alpha, A, A_desc, B, B_desc,
            &beta, C, C_desc, C, C_desc,
            nullptr, nullptr, 0, stream
        );
        
        // Add bias and apply GELU (using custom kernel)
        add_bias_gelu_kernel<<<(M*N + 255)/256, 256, 0, stream>>>(C, bias, M, N);
        
        cublasLtMatrixLayoutDestroy(A_desc);
        cublasLtMatrixLayoutDestroy(B_desc);
        cublasLtMatrixLayoutDestroy(C_desc);
    }
    
    void gemm_bias_baseline(
        const float* A, const float* B, const float* bias, float* C,
        int M, int N, int K, cudaStream_t stream) {
        
        // Setup matrix layouts
        cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_32F, M, K, M);
        cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_32F, K, N, K);
        cublasLtMatrixLayoutCreate(&C_desc, CUDA_R_32F, M, N, M);
        
        // GEMM: C = A @ B
        const float alpha = 1.0f, beta = 0.0f;
        cublasLtMatmul(
            handle, matmul_desc,
            &alpha, A, A_desc, B, B_desc,
            &beta, C, C_desc, C, C_desc,
            nullptr, nullptr, 0, stream
        );
        
        // Add bias
        add_bias_kernel<<<(M*N + 255)/256, 256, 0, stream>>>(C, bias, M, N);
        
        cublasLtMatrixLayoutDestroy(A_desc);
        cublasLtMatrixLayoutDestroy(B_desc);
        cublasLtMatrixLayoutDestroy(C_desc);
    }
};

/**
 * @brief TensorFuse implementation  
 */
class TensorFuseImpl {
public:
    TensorFuseImpl() {
        tensorfuse_simple_init(0); // Initialize with device 0
    }
    
    ~TensorFuseImpl() {
        tensorfuse_simple_cleanup();
    }
    
    float benchmark_ffn_layer(const BertLayer& layer, float* input, float* output, cudaStream_t stream) {
        // Temporary buffer
        float* intermediate;
        cudaMalloc(&intermediate, BATCH_SIZE * SEQ_LEN * FFN_DIM * sizeof(float));
        
        // Time the operations
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start, stream);
        
        // Fused FFN Up: input @ ffn_up_weight + ffn_up_bias + GELU
        fused_gemm_bias_gelu_fp32_wrapper(
            input, layer.ffn_up_weight, layer.ffn_up_bias, intermediate,
            BATCH_SIZE * SEQ_LEN, FFN_DIM, HIDDEN_DIM, 1.0f, 1.0f, stream
        );
        
        // Fused FFN Down: intermediate @ ffn_down_weight + ffn_down_bias (no GELU)
        fused_gemm_bias_gelu_fp32_wrapper(
            intermediate, layer.ffn_down_weight, layer.ffn_down_bias, output,
            BATCH_SIZE * SEQ_LEN, HIDDEN_DIM, FFN_DIM, 1.0f, 1.0f, stream
        );
        
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        
        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(intermediate);
        
        return time_ms;
    }
};

/**
 * @brief Benchmark results structure
 */
struct BenchmarkResults {
    std::string name;
    float avg_time_ms;
    float min_time_ms;
    float max_time_ms;
    float std_dev_ms;
    double tflops;
    double memory_bandwidth_gbps;
    
    void print() const {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "=== " << name << " ===" << std::endl;
        std::cout << "  Average time: " << avg_time_ms << " ms" << std::endl;
        std::cout << "  Min time:     " << min_time_ms << " ms" << std::endl;
        std::cout << "  Max time:     " << max_time_ms << " ms" << std::endl;
        std::cout << "  Std dev:      " << std_dev_ms << " ms" << std::endl;
        std::cout << "  Performance:  " << tflops << " TFLOPS" << std::endl;
        std::cout << "  Bandwidth:    " << memory_bandwidth_gbps << " GB/s" << std::endl;
        std::cout << std::endl;
    }
};

/**
 * @brief Run benchmark for a given implementation
 */
template<typename Implementation>
BenchmarkResults run_benchmark(const std::string& name, Implementation& impl, const BertLayer& layer, float* input, float* output, cudaStream_t stream) {
    std::vector<float> times;
    times.reserve(BENCHMARK_ITERATIONS);
    
    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        impl.benchmark_ffn_layer(layer, input, output, stream);
    }
    
    // Benchmark
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        float time = impl.benchmark_ffn_layer(layer, input, output, stream);
        times.push_back(time);
    }
    
    // Calculate statistics
    float sum = 0.0f;
    float min_time = times[0];
    float max_time = times[0];
    
    for (float time : times) {
        sum += time;
        min_time = std::min(min_time, time);
        max_time = std::max(max_time, time);
    }
    
    float avg_time = sum / times.size();
    
    // Calculate standard deviation
    float variance = 0.0f;
    for (float time : times) {
        variance += (time - avg_time) * (time - avg_time);
    }
    float std_dev = std::sqrt(variance / times.size());
    
    // Calculate TFLOPS and memory bandwidth
    // FFN layer operations: 2 GEMMs + bias + GELU
    long long total_flops = 2LL * BATCH_SIZE * SEQ_LEN * HIDDEN_DIM * FFN_DIM + 
                           2LL * BATCH_SIZE * SEQ_LEN * FFN_DIM * HIDDEN_DIM +
                           2LL * BATCH_SIZE * SEQ_LEN * FFN_DIM; // bias + GELU
    
    double tflops = (total_flops / 1e12) / (avg_time / 1000.0);
    
    // Memory bandwidth estimation
    long long memory_ops = (BATCH_SIZE * SEQ_LEN * HIDDEN_DIM +  // input
                           HIDDEN_DIM * FFN_DIM +                 // weight1
                           FFN_DIM +                              // bias1
                           BATCH_SIZE * SEQ_LEN * FFN_DIM +       // intermediate
                           FFN_DIM * HIDDEN_DIM +                 // weight2
                           HIDDEN_DIM +                           // bias2
                           BATCH_SIZE * SEQ_LEN * HIDDEN_DIM) * sizeof(float); // output
    
    double memory_bandwidth_gbps = (memory_ops / 1e9) / (avg_time / 1000.0);
    
    return {name, avg_time, min_time, max_time, std_dev, tflops, memory_bandwidth_gbps};
}

/**
 * @brief Save results to JSON file
 */
void save_results_json(const std::vector<BenchmarkResults>& results, const std::string& filename) {
    std::ofstream file(filename);
    file << "{\n";
    file << "  \"bert_base_benchmark\": {\n";
    file << "    \"config\": {\n";
    file << "      \"batch_size\": " << BATCH_SIZE << ",\n";
    file << "      \"seq_len\": " << SEQ_LEN << ",\n";
    file << "      \"hidden_dim\": " << HIDDEN_DIM << ",\n";
    file << "      \"ffn_dim\": " << FFN_DIM << ",\n";
    file << "      \"warmup_iterations\": " << WARMUP_ITERATIONS << ",\n";
    file << "      \"benchmark_iterations\": " << BENCHMARK_ITERATIONS << "\n";
    file << "    },\n";
    file << "    \"results\": [\n";
    
    for (size_t i = 0; i < results.size(); i++) {
        const auto& result = results[i];
        file << "      {\n";
        file << "        \"name\": \"" << result.name << "\",\n";
        file << "        \"avg_time_ms\": " << result.avg_time_ms << ",\n";
        file << "        \"min_time_ms\": " << result.min_time_ms << ",\n";
        file << "        \"max_time_ms\": " << result.max_time_ms << ",\n";
        file << "        \"std_dev_ms\": " << result.std_dev_ms << ",\n";
        file << "        \"tflops\": " << result.tflops << ",\n";
        file << "        \"memory_bandwidth_gbps\": " << result.memory_bandwidth_gbps << "\n";
        file << "      }";
        if (i < results.size() - 1) file << ",";
        file << "\n";
    }
    
    file << "    ]\n";
    file << "  }\n";
    file << "}\n";
    file.close();
}

/**
 * @brief Main benchmark function
 */
int main(int argc, char* argv[]) {
    std::cout << "🚀 TensorFuse BERT-base Baseline Benchmark" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    // Print configuration
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size: " << BATCH_SIZE << std::endl;
    std::cout << "  Sequence length: " << SEQ_LEN << std::endl;
    std::cout << "  Hidden dimension: " << HIDDEN_DIM << std::endl;
    std::cout << "  FFN dimension: " << FFN_DIM << std::endl;
    std::cout << "  Warmup iterations: " << WARMUP_ITERATIONS << std::endl;
    std::cout << "  Benchmark iterations: " << BENCHMARK_ITERATIONS << std::endl;
    std::cout << std::endl;
    
    // Initialize CUDA
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Allocate input/output tensors
    float* input;
    float* output;
    cudaMalloc(&input, BATCH_SIZE * SEQ_LEN * HIDDEN_DIM * sizeof(float));
    cudaMalloc(&output, BATCH_SIZE * SEQ_LEN * HIDDEN_DIM * sizeof(float));
    
    // Initialize input with random data
    std::vector<float> host_input(BATCH_SIZE * SEQ_LEN * HIDDEN_DIM);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < host_input.size(); i++) {
        host_input[i] = dist(gen);
    }
    
    cudaMemcpy(input, host_input.data(), host_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create BERT layer
    BertLayer layer;
    
    // Initialize implementations
    CuBLASLtBaseline cublas_impl;
    TensorFuseImpl tensorfuse_impl;
    
    // Run benchmarks
    std::vector<BenchmarkResults> results;
    
    std::cout << "Running cuBLASLt baseline..." << std::endl;
    results.push_back(run_benchmark("cuBLASLt_baseline", cublas_impl, layer, input, output, stream));
    
    std::cout << "Running TensorFuse implementation..." << std::endl;
    results.push_back(run_benchmark("TensorFuse_fused", tensorfuse_impl, layer, input, output, stream));
    
    // Print results
    std::cout << "📊 BENCHMARK RESULTS" << std::endl;
    std::cout << "====================" << std::endl;
    
    for (const auto& result : results) {
        result.print();
    }
    
    // Calculate speedup
    if (results.size() >= 2) {
        float speedup = results[0].avg_time_ms / results[1].avg_time_ms;
        std::cout << "🚀 SPEEDUP: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        std::cout << "🎯 GOAL: 2-7x speedup" << std::endl;
        
        if (speedup >= 2.0f) {
            std::cout << "✅ SUCCESS: Achieved target speedup!" << std::endl;
        } else if (speedup >= 1.5f) {
            std::cout << "⚠️  PARTIAL: Good speedup, room for improvement" << std::endl;
        } else {
            std::cout << "❌ NEEDS WORK: Speedup below target" << std::endl;
        }
    }
    
    // Save results
    save_results_json(results, "benchmarks/results/bert_baseline_results.json");
    std::cout << "📁 Results saved to bert_baseline_results.json" << std::endl;
    
    // Cleanup
    cudaFree(input);
    cudaFree(output);
    cudaStreamDestroy(stream);
    
    return 0;
} 