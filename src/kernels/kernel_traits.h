#pragma once

/**
 * @file kernel_traits.h
 * @brief Common kernel traits and utilities for CUTLASS-based kernels
 */

#include "tensorfuse/types.h"
#include "tensorfuse/config.h"

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/layout.h>
#include <cutlass/numeric_types.h>

// ==============================================================================
// Data Type Mappings
// ==============================================================================

/**
 * @brief Map TensorFuse data types to CUTLASS types
 */
template<TensorFuseDataType dtype>
struct TensorFuseDataTypeTraits;

template<>
struct TensorFuseDataTypeTraits<TENSORFUSE_FLOAT32> {
    using type = float;
    static constexpr int alignment = 4;
};

template<>
struct TensorFuseDataTypeTraits<TENSORFUSE_FLOAT16> {
    using type = cutlass::half_t;
    static constexpr int alignment = 8;
};

template<>
struct TensorFuseDataTypeTraits<TENSORFUSE_BFLOAT16> {
    using type = cutlass::bfloat16_t;
    static constexpr int alignment = 8;
};

template<>
struct TensorFuseDataTypeTraits<TENSORFUSE_INT8> {
    using type = int8_t;
    static constexpr int alignment = 16;
};

template<>
struct TensorFuseDataTypeTraits<TENSORFUSE_UINT8> {
    using type = uint8_t;
    static constexpr int alignment = 16;
};

template<>
struct TensorFuseDataTypeTraits<TENSORFUSE_INT32> {
    using type = int32_t;
    static constexpr int alignment = 4;
};

#if TENSORFUSE_HAS_FP8_SUPPORT
template<>
struct TensorFuseDataTypeTraits<TENSORFUSE_FP8_E4M3> {
    using type = cutlass::float_e4m3_t;
    static constexpr int alignment = 16;
};

template<>
struct TensorFuseDataTypeTraits<TENSORFUSE_FP8_E5M2> {
    using type = cutlass::float_e5m2_t;
    static constexpr int alignment = 16;
};
#endif

// ==============================================================================
// Architecture Detection
// ==============================================================================

/**
 * @brief Architecture traits for different compute capabilities
 */
template<int Major, int Minor>
struct ArchTraits;

template<>
struct ArchTraits<8, 0> {
    using Arch = cutlass::arch::Sm80;
    static constexpr bool supports_tensor_cores = true;
    static constexpr bool supports_fp16_tc = true;
    static constexpr bool supports_bf16_tc = true;
    static constexpr bool supports_fp8_tc = false;
    static constexpr int max_shared_memory_kb = 48;
};

template<>
struct ArchTraits<8, 6> {
    using Arch = cutlass::arch::Sm86;
    static constexpr bool supports_tensor_cores = true;
    static constexpr bool supports_fp16_tc = true;
    static constexpr bool supports_bf16_tc = true;
    static constexpr bool supports_fp8_tc = false;
    static constexpr int max_shared_memory_kb = 48;
};

template<>
struct ArchTraits<8, 9> {
    using Arch = cutlass::arch::Sm89;
    static constexpr bool supports_tensor_cores = true;
    static constexpr bool supports_fp16_tc = true;
    static constexpr bool supports_bf16_tc = true;
    static constexpr bool supports_fp8_tc = false;
    static constexpr int max_shared_memory_kb = 48;
};

template<>
struct ArchTraits<9, 0> {
    using Arch = cutlass::arch::Sm90;
    static constexpr bool supports_tensor_cores = true;
    static constexpr bool supports_fp16_tc = true;
    static constexpr bool supports_bf16_tc = true;
    static constexpr bool supports_fp8_tc = true;
    static constexpr int max_shared_memory_kb = 48;
};

// ==============================================================================
// Kernel Configuration Traits
// ==============================================================================

/**
 * @brief Default kernel configuration for different problem sizes
 */
template<int M, int N, int K>
struct DefaultKernelConfig {
    // Small problems (M*N*K < 1M operations)
    static constexpr int tile_M = (M < 64) ? 32 : 64;
    static constexpr int tile_N = (N < 64) ? 32 : 64;
    static constexpr int tile_K = (K < 64) ? 32 : 64;
    static constexpr int warp_M = tile_M / 2;
    static constexpr int warp_N = tile_N / 2;
    static constexpr int warp_K = tile_K / 2;
    static constexpr int stages = 2;
};

/**
 * @brief Optimized kernel configuration for medium problems
 */
template<>
struct DefaultKernelConfig<128, 128, 128> {
    static constexpr int tile_M = 128;
    static constexpr int tile_N = 128;
    static constexpr int tile_K = 64;
    static constexpr int warp_M = 64;
    static constexpr int warp_N = 64;
    static constexpr int warp_K = 32;
    static constexpr int stages = 3;
};

/**
 * @brief Optimized kernel configuration for large problems
 */
template<>
struct DefaultKernelConfig<256, 256, 256> {
    static constexpr int tile_M = 256;
    static constexpr int tile_N = 128;
    static constexpr int tile_K = 64;
    static constexpr int warp_M = 64;
    static constexpr int warp_N = 64;
    static constexpr int warp_K = 32;
    static constexpr int stages = 4;
};

// ==============================================================================
// Performance Estimation
// ==============================================================================

/**
 * @brief Estimate kernel performance parameters
 */
struct KernelPerformanceEstimate {
    float estimated_time_ms;
    float estimated_flops;
    float estimated_bandwidth_gb_s;
    float tensor_core_utilization;
    float memory_efficiency;
    int optimal_occupancy;
};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Get theoretical peak performance for device
 */
TensorFuseStatus tensorfuse_get_device_peak_performance(
    int device_id,
    float* peak_flops,
    float* peak_bandwidth_gb_s,
    int* compute_capability_major,
    int* compute_capability_minor
);

/**
 * @brief Estimate GEMM kernel performance
 */
TensorFuseStatus tensorfuse_estimate_gemm_performance(
    int M, int N, int K,
    int tile_M, int tile_N, int tile_K,
    TensorFuseDataType input_dtype,
    TensorFuseDataType weight_dtype,
    TensorFuseDataType output_dtype,
    int device_id,
    KernelPerformanceEstimate* estimate
);

// ==============================================================================
// Memory Layout Utilities
// ==============================================================================

/**
 * @brief Get optimal memory layout for tensor
 */
TensorFuseStatus tensorfuse_get_optimal_layout(
    const TensorFuseShape* shape,
    TensorFuseDataType dtype,
    bool is_weight,
    TensorFuseLayout* optimal_layout
);

/**
 * @brief Calculate memory alignment requirements
 */
TensorFuseStatus tensorfuse_calculate_memory_alignment(
    TensorFuseDataType dtype,
    int* alignment_bytes
);

/**
 * @brief Calculate optimal stride for tensor
 */
TensorFuseStatus tensorfuse_calculate_optimal_stride(
    const TensorFuseShape* shape,
    TensorFuseDataType dtype,
    TensorFuseLayout layout,
    int* strides
);

// ==============================================================================
// Kernel Selection
// ==============================================================================

/**
 * @brief Select best kernel configuration for problem
 */
TensorFuseStatus tensorfuse_select_kernel_config(
    int M, int N, int K,
    TensorFuseDataType input_dtype,
    TensorFuseDataType weight_dtype,
    TensorFuseDataType output_dtype,
    int device_id,
    TensorFuseKernelConfig* config
);

/**
 * @brief Check if kernel configuration is valid for device
 */
TensorFuseStatus tensorfuse_validate_kernel_config(
    const TensorFuseKernelConfig* config,
    int device_id,
    bool* is_valid
);

// ==============================================================================
// Shared Memory Utilities
// ==============================================================================

/**
 * @brief Calculate shared memory requirements for kernel
 */
TensorFuseStatus tensorfuse_calculate_shared_memory_usage(
    const TensorFuseKernelConfig* config,
    TensorFuseDataType dtype,
    size_t* shared_memory_bytes
);

/**
 * @brief Get maximum shared memory per block for device
 */
TensorFuseStatus tensorfuse_get_max_shared_memory_per_block(
    int device_id,
    size_t* max_shared_memory_bytes
);

// ==============================================================================
// Tensor Core Utilities
// ==============================================================================

/**
 * @brief Check if data types support Tensor Cores
 */
bool tensorfuse_supports_tensor_cores(
    TensorFuseDataType input_dtype,
    TensorFuseDataType weight_dtype,
    TensorFuseDataType output_dtype,
    int device_id
);

/**
 * @brief Get Tensor Core instruction shape
 */
TensorFuseStatus tensorfuse_get_tensor_core_instruction_shape(
    TensorFuseDataType input_dtype,
    TensorFuseDataType weight_dtype,
    int device_id,
    int* instruction_M,
    int* instruction_N,
    int* instruction_K
);

/**
 * @brief Calculate optimal warp configuration for Tensor Cores
 */
TensorFuseStatus tensorfuse_get_optimal_warp_config(
    int tile_M, int tile_N, int tile_K,
    TensorFuseDataType dtype,
    int device_id,
    int* warp_M, int* warp_N, int* warp_K
);

#ifdef __cplusplus
} // extern "C"

// ==============================================================================
// C++ Template Utilities
// ==============================================================================

namespace tensorfuse {
namespace kernels {

/**
 * @brief CUTLASS type mapping helper
 */
template<TensorFuseDataType dtype>
using CutlassType = typename TensorFuseDataTypeTraits<dtype>::type;

/**
 * @brief Get CUTLASS layout from TensorFuse layout
 */
template<TensorFuseLayout layout>
struct LayoutTraits;

template<>
struct LayoutTraits<TENSORFUSE_LAYOUT_ROW_MAJOR> {
    using type = cutlass::layout::RowMajor;
};

template<>
struct LayoutTraits<TENSORFUSE_LAYOUT_COL_MAJOR> {
    using type = cutlass::layout::ColumnMajor;
};

/**
 * @brief Kernel configuration selector
 */
template<typename InputType, typename WeightType, typename OutputType, typename ArchTag>
struct KernelConfigSelector {
    // Default configuration - can be specialized for specific combinations
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    static constexpr int Stages = 3;
    static constexpr int Alignment = 8;
};

/**
 * @brief Performance heuristics
 */
template<typename InputType, typename WeightType, typename OutputType>
struct PerformanceHeuristics {
    static constexpr bool prefer_split_k(int M, int N, int K) {
        return K > 1024 && M * N < 4096;
    }
    
    static constexpr int optimal_stages(int shared_memory_kb) {
        return (shared_memory_kb >= 48) ? 4 : 3;
    }
    
    static constexpr bool use_warp_specialization(int M, int N) {
        return M >= 128 && N >= 128;
    }
};

} // namespace kernels
} // namespace tensorfuse

#endif // __cplusplus 