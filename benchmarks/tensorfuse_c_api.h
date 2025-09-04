/**
 * @file tensorfuse_c_api.h
 * @brief Standalone TensorFuse wrapper functions for benchmarks
 * 
 * This header provides direct access to wrapper functions without
 * any dependencies on the main TensorFuse headers to avoid conflicts.
 */

#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief TensorFuse status codes (standalone definition)
 */
typedef enum {
    TENSORFUSE_SUCCESS = 0,
    TENSORFUSE_CUDA_ERROR,
    TENSORFUSE_CUTLASS_ERROR,
    TENSORFUSE_INVALID_ARGUMENT,
    TENSORFUSE_UNSUPPORTED_ARCH,
    TENSORFUSE_OUT_OF_MEMORY,
    TENSORFUSE_NOT_INITIALIZED
} TensorFuseStatus;

/**
 * @brief Initialize TensorFuse with default configuration
 * @return TensorFuseStatus indicating success or failure
 */
TensorFuseStatus tensorfuse_simple_init(int device_id);

/**
 * @brief Cleanup TensorFuse
 */
void tensorfuse_simple_cleanup(void);

/**
 * @brief FP32 Wrapper for fused GEMM + Bias + GELU
 */
TensorFuseStatus fused_gemm_bias_gelu_fp32_wrapper(
    const void* A, const void* B, const void* bias, void* C,
    int M, int N, int K, float alpha, float beta, cudaStream_t stream);

/**
 * @brief FP16 Wrapper for fused GEMM + Bias + GELU
 */
TensorFuseStatus fused_gemm_bias_gelu_fp16_wrapper(
    const void* A, const void* B, const void* bias, void* C,
    int M, int N, int K, float alpha, float beta, cudaStream_t stream);

/**
 * @brief INT8 Wrapper for fused GEMM + Bias + GELU
 */
TensorFuseStatus fused_gemm_bias_gelu_int8_wrapper(
    const void* A, const void* B, const void* bias, void* C,
    int M, int N, int K, float alpha, float beta, cudaStream_t stream);

#ifdef __cplusplus
}
#endif 