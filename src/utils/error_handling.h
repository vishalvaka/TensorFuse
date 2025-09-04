/*
 * Copyright (c) 2024, TensorFuse Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Error handling utilities and status codes
 */

#pragma once

#include "tensorfuse/types.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <string>

namespace tensorfuse {

/**
 * @brief Convert TensorFuseStatus to string
 * @param status Status code
 * @return Human-readable error description
 */
const char* get_error_string(TensorFuseStatus status);

/**
 * @brief Convert CUDA error to TensorFuseStatus
 * @param error CUDA error code
 * @return Corresponding TensorFuseStatus
 */
TensorFuseStatus cuda_error_to_status(cudaError_t error);

/**
 * @brief Convert cuBLAS error to TensorFuseStatus
 * @param error cuBLAS error code
 * @return Corresponding TensorFuseStatus
 */
TensorFuseStatus cublas_error_to_status(cublasStatus_t error);

/**
 * @brief Convert cuDNN error to TensorFuseStatus
 * @param error cuDNN error code
 * @return Corresponding TensorFuseStatus
 */
TensorFuseStatus cudnn_error_to_status(cudnnStatus_t error);

/**
 * @brief Convert cuRAND error to TensorFuseStatus
 * @param error cuRAND error code
 * @return Corresponding TensorFuseStatus
 */
TensorFuseStatus curand_error_to_status(curandStatus_t error);

/**
 * @brief Check if status indicates success
 * @param status Status code to check
 * @return true if successful
 */
inline bool is_success(TensorFuseStatus status) {
    return status == TENSORFUSE_SUCCESS;
}

/**
 * @brief Check if status indicates failure
 * @param status Status code to check
 * @return true if failed
 */
inline bool is_failure(TensorFuseStatus status) {
    return status != TENSORFUSE_SUCCESS;
}

} // namespace tensorfuse

// Error checking macros
#define TENSORFUSE_CHECK_CUDA(call) \
    do { \
        cudaError_t error = (call); \
        if (error != cudaSuccess) { \
            return ::tensorfuse::cuda_error_to_status(error); \
        } \
    } while (0)

#define TENSORFUSE_CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t error = (call); \
        if (error != CUBLAS_STATUS_SUCCESS) { \
            return ::tensorfuse::cublas_error_to_status(error); \
        } \
    } while (0)

#define TENSORFUSE_CHECK_CUDNN(call) \
    do { \
        cudnnStatus_t error = (call); \
        if (error != CUDNN_STATUS_SUCCESS) { \
            return ::tensorfuse::cudnn_error_to_status(error); \
        } \
    } while (0)

#define TENSORFUSE_CHECK_CURAND(call) \
    do { \
        curandStatus_t error = (call); \
        if (error != CURAND_STATUS_SUCCESS) { \
            return ::tensorfuse::curand_error_to_status(error); \
        } \
    } while (0)

#define TENSORFUSE_CHECK_STATUS(call) \
    do { \
        TensorFuseStatus status = (call); \
        if (::tensorfuse::is_failure(status)) { \
            return status; \
        } \
    } while (0)
