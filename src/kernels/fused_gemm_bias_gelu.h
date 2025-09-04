/*
 * Copyright (c) 2024, TensorFuse Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Fused GEMM+Bias+GELU kernel declarations
 */

#pragma once

#include "tensorfuse/types.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// ==============================================================================
// FP32 Kernels
// ==============================================================================

/**
 * @brief Fused GEMM + Bias + GELU operation using FP32
 * 
 * @param A Input matrix A [M x K]
 * @param B Weight matrix B [K x N]  
 * @param bias Bias vector [N]
 * @param C Output matrix C [M x N]
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param alpha Scaling factor for A*B
 * @param beta Scaling factor for bias
 * @param stream CUDA stream
 * @return Status code
 */
TensorFuseStatus fused_gemm_bias_gelu_fp32(
    const void* A, const void* B, const void* bias, void* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream);

// ==============================================================================
// FP16 Kernels
// ==============================================================================

/**
 * @brief Fused GEMM + Bias + GELU operation using FP16
 * 
 * @param A Input matrix A [M x K]
 * @param B Weight matrix B [K x N]  
 * @param bias Bias vector [N]
 * @param C Output matrix C [M x N]
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param alpha Scaling factor for A*B
 * @param beta Scaling factor for bias
 * @param stream CUDA stream
 * @return Status code
 */
TensorFuseStatus fused_gemm_bias_gelu_fp16(
    const void* A, const void* B, const void* bias, void* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream);

// ==============================================================================
// INT8 Tensor Core Kernels - The Performance Multiplier!
// ==============================================================================

/**
 * @brief Fused GEMM + Bias + GELU operation using INT8 Tensor Cores
 * 
 * This is the critical kernel for achieving 2-7x speedup on Ada Lovelace GPUs
 * 
 * @param A Input matrix A [M x K] - INT8 quantized
 * @param B Weight matrix B [K x N] - INT8 quantized
 * @param bias Bias vector [N] - FP32 for precision
 * @param C Output matrix C [M x N] - FP32 for precision
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param alpha Scaling factor for A*B
 * @param beta Scaling factor for bias
 * @param stream CUDA stream
 * @return Status code
 */
TensorFuseStatus fused_gemm_bias_gelu_int8(
    const void* A, const void* B, const void* bias, void* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream);

// ==============================================================================
// C API Wrappers
// ==============================================================================

TensorFuseStatus fused_gemm_bias_gelu_fp32_wrapper(
    const void* A, const void* B, const void* bias, void* C,
    int M, int N, int K, float alpha, float beta, cudaStream_t stream);

TensorFuseStatus fused_gemm_bias_gelu_fp16_wrapper(
    const void* A, const void* B, const void* bias, void* C,
    int M, int N, int K, float alpha, float beta, cudaStream_t stream);

TensorFuseStatus fused_gemm_bias_gelu_int8_wrapper(
    const void* A, const void* B, const void* bias, void* C,
    int M, int N, int K, float alpha, float beta, cudaStream_t stream);

#ifdef __cplusplus
}
#endif 