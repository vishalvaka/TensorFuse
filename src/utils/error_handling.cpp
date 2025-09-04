/*
 * Copyright (c) 2024, TensorFuse Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Error handling utilities and status code conversions
 */

#include "utils/error_handling.h"

namespace tensorfuse {

const char* get_error_string(TensorFuseStatus status) {
    switch (status) {
        case TENSORFUSE_SUCCESS:
            return "Success";
        case TENSORFUSE_ERROR_INVALID_ARGUMENT:
            return "Invalid argument";
        case TENSORFUSE_ERROR_CUDA_ERROR:
            return "CUDA error";
        case TENSORFUSE_ERROR_OUT_OF_MEMORY:
            return "Out of memory";
        case TENSORFUSE_ERROR_NOT_INITIALIZED:
            return "Not initialized";
        case TENSORFUSE_ERROR_INVALID_CONFIGURATION:
            return "Invalid configuration";
        case TENSORFUSE_ERROR_KERNEL_LAUNCH_FAILED:
            return "Kernel launch failed";
        case TENSORFUSE_ERROR_AUTOTUNER_FAILED:
            return "Autotuner failed";
        case TENSORFUSE_ERROR_PROFILING_FAILED:
            return "Profiling failed";
        case TENSORFUSE_ERROR_FILE_IO:
            return "File I/O error";
        case TENSORFUSE_ERROR_UNSUPPORTED_OPERATION:
            return "Unsupported operation";
        case TENSORFUSE_ERROR_INVALID_TENSOR_SHAPE:
            return "Invalid tensor shape";
        case TENSORFUSE_ERROR_UNKNOWN:
        default:
            return "Unknown error";
    }
}

TensorFuseStatus cuda_error_to_status(cudaError_t error) {
    switch (error) {
        case cudaSuccess:
            return TENSORFUSE_SUCCESS;
        case cudaErrorInvalidValue:
        case cudaErrorInvalidConfiguration:
        case cudaErrorInvalidDevice:
        case cudaErrorInvalidDevicePointer:
        case cudaErrorInvalidSymbol:
        case cudaErrorInvalidTexture:
        case cudaErrorInvalidTextureBinding:
        case cudaErrorInvalidChannelDescriptor:
        case cudaErrorInvalidMemcpyDirection:
        case cudaErrorInvalidResourceHandle:
        case cudaErrorInvalidPitchValue:
        case cudaErrorInvalidFilterSetting:
        case cudaErrorInvalidNormSetting:
            return TENSORFUSE_ERROR_INVALID_ARGUMENT;
        case cudaErrorMemoryAllocation:
            return TENSORFUSE_ERROR_OUT_OF_MEMORY;
        case cudaErrorNotReady:
        case cudaErrorNotYetImplemented:
        case cudaErrorNoDevice:
        case cudaErrorInsufficientDriver:
        case cudaErrorSetOnActiveProcess:
        case cudaErrorInvalidSurface:
        case cudaErrorNoKernelImageForDevice:
        case cudaErrorIncompatibleDriverContext:
        case cudaErrorPeerAccessAlreadyEnabled:
        case cudaErrorPeerAccessNotEnabled:
        case cudaErrorDeviceAlreadyInUse:
        case cudaErrorProfilerDisabled:
        case cudaErrorProfilerNotInitialized:
        case cudaErrorProfilerAlreadyStarted:
        case cudaErrorProfilerAlreadyStopped:
        case cudaErrorAssert:
        case cudaErrorTooManyPeers:
        case cudaErrorHostMemoryAlreadyRegistered:
        case cudaErrorHostMemoryNotRegistered:
        case cudaErrorOperatingSystem:
        case cudaErrorPeerAccessUnsupported:
        case cudaErrorLaunchFailure:
        case cudaErrorLaunchTimeout:
        case cudaErrorLaunchOutOfResources:
        case cudaErrorInvalidDeviceFunction:
        case cudaErrorMissingConfiguration:
        case cudaErrorPriorLaunchFailure:
        case cudaErrorInvalidAddressSpace:
        case cudaErrorInvalidPc:
        case cudaErrorIllegalAddress:
        case cudaErrorIllegalInstruction:
        case cudaErrorMisalignedAddress:
        case cudaErrorHardwareStackError:
        case cudaErrorSystemDriverMismatch:
        case cudaErrorCompatNotSupportedOnDevice:
        default:
            return TENSORFUSE_ERROR_CUDA_ERROR;
    }
}

TensorFuseStatus cublas_error_to_status(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return TENSORFUSE_SUCCESS;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return TENSORFUSE_ERROR_NOT_INITIALIZED;
        case CUBLAS_STATUS_ALLOC_FAILED:
            return TENSORFUSE_ERROR_OUT_OF_MEMORY;
        case CUBLAS_STATUS_INVALID_VALUE:
            return TENSORFUSE_ERROR_INVALID_ARGUMENT;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return TENSORFUSE_ERROR_INVALID_CONFIGURATION;
        case CUBLAS_STATUS_MAPPING_ERROR:
        case CUBLAS_STATUS_EXECUTION_FAILED:
        case CUBLAS_STATUS_INTERNAL_ERROR:
        case CUBLAS_STATUS_NOT_SUPPORTED:
        case CUBLAS_STATUS_LICENSE_ERROR:
        default:
            return TENSORFUSE_ERROR_CUDA_ERROR;
    }
}

TensorFuseStatus cudnn_error_to_status(cudnnStatus_t error) {
    switch (error) {
        case CUDNN_STATUS_SUCCESS:
            return TENSORFUSE_SUCCESS;
        case CUDNN_STATUS_NOT_INITIALIZED:
            return TENSORFUSE_ERROR_NOT_INITIALIZED;
        case CUDNN_STATUS_ALLOC_FAILED:
            return TENSORFUSE_ERROR_OUT_OF_MEMORY;
        case CUDNN_STATUS_BAD_PARAM:
            return TENSORFUSE_ERROR_INVALID_ARGUMENT;
        case CUDNN_STATUS_ARCH_MISMATCH:
            return TENSORFUSE_ERROR_INVALID_CONFIGURATION;
        case CUDNN_STATUS_MAPPING_ERROR:
        case CUDNN_STATUS_EXECUTION_FAILED:
        case CUDNN_STATUS_INTERNAL_ERROR:
        case CUDNN_STATUS_NOT_SUPPORTED:
        case CUDNN_STATUS_LICENSE_ERROR:
        case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
        case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
        case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
        default:
            return TENSORFUSE_ERROR_CUDA_ERROR;
    }
}

TensorFuseStatus curand_error_to_status(curandStatus_t error) {
    switch (error) {
        case CURAND_STATUS_SUCCESS:
            return TENSORFUSE_SUCCESS;
        case CURAND_STATUS_VERSION_MISMATCH:
            return TENSORFUSE_ERROR_INVALID_CONFIGURATION;
        case CURAND_STATUS_NOT_INITIALIZED:
            return TENSORFUSE_ERROR_NOT_INITIALIZED;
        case CURAND_STATUS_ALLOCATION_FAILED:
            return TENSORFUSE_ERROR_OUT_OF_MEMORY;
        case CURAND_STATUS_TYPE_ERROR:
        case CURAND_STATUS_OUT_OF_RANGE:
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return TENSORFUSE_ERROR_INVALID_ARGUMENT;
        case CURAND_STATUS_LAUNCH_FAILURE:
            return TENSORFUSE_ERROR_KERNEL_LAUNCH_FAILED;
        case CURAND_STATUS_PREEXISTING_FAILURE:
        case CURAND_STATUS_INITIALIZATION_FAILED:
        case CURAND_STATUS_ARCH_MISMATCH:
        case CURAND_STATUS_INTERNAL_ERROR:
        default:
            return TENSORFUSE_ERROR_CUDA_ERROR;
    }
}

} // namespace tensorfuse 