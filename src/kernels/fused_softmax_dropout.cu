/*
 * Copyright (c) 2024, TensorFuse Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Fused Softmax+Dropout kernel implementation optimized for attention computation
 */

#include "kernels/fused_softmax_dropout.h"
#include "utils/error_handling.h"
#include "utils/cuda_utils.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>

namespace tensorfuse {
namespace kernels {

// Warp-level primitives for efficient reduction
template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    constexpr int kWarpSize = 32;
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    constexpr int kWarpSize = 32;
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Online softmax kernel with dropout
template <typename T, int kBlockSize>
__global__ void fused_softmax_dropout_kernel(
    const T* input,
    T* output,
    unsigned char* dropout_mask,
    const float dropout_prob,
    const unsigned long long seed,
    const int batch_size,
    const int seq_len,
    const int head_dim,
    const int num_heads) {

    constexpr int kWarpSize = 32;
    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane_id = tid % kWarpSize;
    const int num_warps = kBlockSize / kWarpSize;
    
    // Block-level shared memory
    __shared__ float shared_max[kBlockSize / kWarpSize];
    __shared__ float shared_sum[kBlockSize / kWarpSize];
    
    // Calculate global indices
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.z;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len) {
        return;
    }
    
    const int row_offset = ((batch_idx * num_heads + head_idx) * seq_len + seq_idx) * head_dim;
    
    // Initialize curand state
    curandState_t state;
    curand_init(seed, blockIdx.x * blockDim.x + threadIdx.x, 0, &state);
    
    // Online softmax computation
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;
    
    // First pass: compute max
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float val = static_cast<float>(input[row_offset + i]);
        thread_max = fmaxf(thread_max, val);
    }
    
    // Reduce max across warp
    float warp_max = warp_reduce_max(thread_max);
    
    // Store warp max in shared memory
    if (lane_id == 0) {
        shared_max[warp_id] = warp_max;
    }
    __syncthreads();
    
    // Reduce across warps
    float block_max = -INFINITY;
    if (tid < num_warps) {
        block_max = shared_max[tid];
    }
    if (warp_id == 0) {
        block_max = warp_reduce_max(block_max);
        if (lane_id == 0) {
            shared_max[0] = block_max;
        }
    }
    __syncthreads();
    block_max = shared_max[0];
    
    // Second pass: compute sum of exp(x - max)
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float val = static_cast<float>(input[row_offset + i]);
        float exp_val = expf(val - block_max);
        thread_sum += exp_val;
    }
    
    // Reduce sum across warp
    float warp_sum = warp_reduce_sum(thread_sum);
    
    // Store warp sum in shared memory
    if (lane_id == 0) {
        shared_sum[warp_id] = warp_sum;
    }
    __syncthreads();
    
    // Reduce across warps
    float block_sum = 0.0f;
    if (tid < num_warps) {
        block_sum = shared_sum[tid];
    }
    if (warp_id == 0) {
        block_sum = warp_reduce_sum(block_sum);
        if (lane_id == 0) {
            shared_sum[0] = block_sum;
        }
    }
    __syncthreads();
    block_sum = shared_sum[0];
    
    // Third pass: compute softmax and apply dropout
    const float inv_sum = 1.0f / block_sum;
    const float dropout_scale = 1.0f / (1.0f - dropout_prob);
    
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float val = static_cast<float>(input[row_offset + i]);
        float softmax_val = expf(val - block_max) * inv_sum;
        
        // Apply dropout
        float random_val = curand_uniform(&state);
        bool keep = random_val > dropout_prob;
        
        // Store dropout mask
        dropout_mask[row_offset + i] = keep ? 1 : 0;
        
        // Apply dropout and scale
        float output_val = keep ? (softmax_val * dropout_scale) : 0.0f;
        output[row_offset + i] = static_cast<T>(output_val);
    }
}

// Optimized kernel for small sequence lengths (warp-level)
template <typename T>
__global__ void fused_softmax_dropout_warp_kernel(
    const T* input,
    T* output,
    unsigned char* dropout_mask,
    const float dropout_prob,
    const unsigned long long seed,
    const int batch_size,
    const int seq_len,
    const int head_dim,
    const int num_heads) {
    
    const int lane_id = threadIdx.x % 32;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    
    // Calculate which sequence this warp processes
    const int total_sequences = batch_size * num_heads * seq_len;
    if (warp_id >= total_sequences) return;
    
    const int batch_idx = warp_id / (num_heads * seq_len);
    const int remaining = warp_id % (num_heads * seq_len);
    const int head_idx = remaining / seq_len;
    const int seq_idx = remaining % seq_len;
    
    const int row_offset = ((batch_idx * num_heads + head_idx) * seq_len + seq_idx) * head_dim;
    
    // Initialize curand state
    curandState_t state;
    curand_init(seed, warp_id * 32 + lane_id, 0, &state);
    
    // Each warp processes one sequence
    
    // Simplified implementation: Only use the first thread in each warp
    // This avoids issues with warp reductions when not all threads are active
    if (lane_id == 0) {
        // Process entire sequence in first thread
        float max_val = -INFINITY;
        
        // Find max for numerical stability
        for (int i = 0; i < head_dim; i++) {
            float val = static_cast<float>(input[row_offset + i]);
            max_val = fmaxf(max_val, val);
        }
        
        // Compute sum of exp(x - max)
        float sum_exp = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            float val = static_cast<float>(input[row_offset + i]);
            float exp_val = expf(val - max_val);
            sum_exp += exp_val;
        }
        
        // Compute softmax and apply dropout
        const float inv_sum = 1.0f / sum_exp;
        const float dropout_scale = 1.0f / (1.0f - dropout_prob);
        
        for (int i = 0; i < head_dim; i++) {
            float val = static_cast<float>(input[row_offset + i]);
            float softmax_val = expf(val - max_val) * inv_sum;
            
            // Apply dropout - fix the edge case for dropout_prob = 0.0
            float random_val = curand_uniform(&state);
            bool keep = (dropout_prob == 0.0f) || (random_val > dropout_prob);
            
            // Store dropout mask
            dropout_mask[row_offset + i] = keep ? 1 : 0;
            
            // Apply dropout and scale
            float output_val = keep ? (softmax_val * dropout_scale) : 0.0f;
            output[row_offset + i] = static_cast<T>(output_val);
        }
    }
}

// Kernel launcher
template <typename T>
TensorFuseStatus launch_fused_softmax_dropout_kernel(
    const void* input,
    void* output,
    unsigned char* dropout_mask,
    int batch_size,
    int seq_len,
    int head_dim,
    int num_heads,
    float dropout_prob,
    unsigned long long seed,
    cudaStream_t stream) {
    
    const T* input_ptr = static_cast<const T*>(input);
    T* output_ptr = static_cast<T*>(output);
    
    // Choose kernel based on head dimension (softmax axis)
    if (head_dim <= 1024) {
        // Use warp-level kernel for small dimensions
        const int threads_per_block = 256;
        constexpr int kWarpSize = 32;
        const int warps_per_block = threads_per_block / kWarpSize;
        const int total_sequences = batch_size * num_heads * seq_len;
        const int blocks = (total_sequences + warps_per_block - 1) / warps_per_block;
        
        fused_softmax_dropout_warp_kernel<T><<<blocks, threads_per_block, 0, stream>>>(
            input_ptr, output_ptr, dropout_mask, dropout_prob, seed,
            batch_size, seq_len, head_dim, num_heads);
    } else {
        // Use block-level kernel for larger dimensions
        const int threads_per_block = 256;
        dim3 grid_dim(batch_size, num_heads, seq_len);
        
        fused_softmax_dropout_kernel<T, threads_per_block><<<grid_dim, threads_per_block, 0, stream>>>(
            input_ptr, output_ptr, dropout_mask, dropout_prob, seed,
            batch_size, seq_len, head_dim, num_heads);
    }
    
    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        return TENSORFUSE_ERROR_CUDA_ERROR;
    }
    
    return TENSORFUSE_SUCCESS;
}

// Public API implementations
TensorFuseStatus fused_softmax_dropout_fp16(
    const void* input,
    void* output,
    unsigned char* dropout_mask,
    int batch_size,
    int seq_len,
    int head_dim,
    int num_heads,
    float dropout_prob,
    unsigned long long seed,
    cudaStream_t stream) {
    
    return launch_fused_softmax_dropout_kernel<half>(
        input, output, dropout_mask, batch_size, seq_len, head_dim, num_heads,
        dropout_prob, seed, stream);
}

TensorFuseStatus fused_softmax_dropout_fp32(
    const void* input,
    void* output,
    unsigned char* dropout_mask,
    int batch_size,
    int seq_len,
    int head_dim,
    int num_heads,
    float dropout_prob,
    unsigned long long seed,
    cudaStream_t stream) {
    
    return launch_fused_softmax_dropout_kernel<float>(
        input, output, dropout_mask, batch_size, seq_len, head_dim, num_heads,
        dropout_prob, seed, stream);
}

} // namespace kernels
} // namespace tensorfuse 

// C wrapper functions for C API
extern "C" {

TensorFuseStatus fused_softmax_dropout_fp16_wrapper(
    const void* input, void* output, unsigned char* dropout_mask,
    int batch_size, int seq_len, int head_dim, int num_heads,
    float dropout_prob, unsigned long long seed, cudaStream_t stream) {
    
    return tensorfuse::kernels::fused_softmax_dropout_fp16(
        input, output, dropout_mask, batch_size, seq_len, head_dim, num_heads,
        dropout_prob, seed, stream);
}

TensorFuseStatus fused_softmax_dropout_fp32_wrapper(
    const void* input, void* output, unsigned char* dropout_mask,
    int batch_size, int seq_len, int head_dim, int num_heads,
    float dropout_prob, unsigned long long seed, cudaStream_t stream) {
    
    return tensorfuse::kernels::fused_softmax_dropout_fp32(
        input, output, dropout_mask, batch_size, seq_len, head_dim, num_heads,
        dropout_prob, seed, stream);
}

} // extern "C"