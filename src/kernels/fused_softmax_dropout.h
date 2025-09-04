#pragma once

/**
 * @file fused_softmax_dropout.h
 * @brief Fused Softmax + Dropout kernel implementation
 */

#include "tensorfuse/types.h"
#include <cuda_runtime.h>

namespace tensorfuse {
namespace kernels {

/**
 * @brief Fused Softmax + Dropout kernel for FP16 data
 * @param input Input tensor
 * @param output Output tensor
 * @param dropout_mask Dropout mask output
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param head_dim Head dimension
 * @param num_heads Number of heads
 * @param dropout_prob Dropout probability
 * @param seed Random seed
 * @param stream CUDA stream
 * @return TensorFuseStatus indicating success or failure
 */
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
    cudaStream_t stream
);

/**
 * @brief Fused Softmax + Dropout kernel for FP32 data
 * @param input Input tensor
 * @param output Output tensor
 * @param dropout_mask Dropout mask output
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param head_dim Head dimension
 * @param num_heads Number of heads
 * @param dropout_prob Dropout probability
 * @param seed Random seed
 * @param stream CUDA stream
 * @return TensorFuseStatus indicating success or failure
 */
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
    cudaStream_t stream
);

} // namespace kernels
} // namespace tensorfuse 