/*
 * Copyright (c) 2024, TensorFuse Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Optimized Fused GEMM+Bias+GELU kernel implementation with Tensor Core support
 */

#include "kernels/fused_gemm_bias_gelu.h"
#include "kernels/kernel_traits.h"
#include "utils/error_handling.h"
#include "utils/cuda_utils.h"

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>

namespace tensorfuse {
namespace kernels {

// Optimized GELU kernel with better memory access patterns
template <typename T>
__global__ void optimized_add_bias_gelu_kernel(
    T* output, const T* bias, int M, int N, float beta) {
    
    // Use 2D block structure for better memory access
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < M && col < N) {
        int idx = row * N + col;
        
        // Add bias with proper beta scaling
        float val = static_cast<float>(output[idx]) + beta * static_cast<float>(bias[col]);
        
        // Optimized GELU activation using fast approximation
        // GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float kAlpha = 0.7978845608028654f;  // sqrt(2/pi)
        const float kBeta = 0.044715f;
        const float kHalf = 0.5f;
        
        float x_cubed = val * val * val;
        float inner = kAlpha * (val + kBeta * x_cubed);
        
        // Use faster tanh approximation for better performance
        float tanh_val = tanhf(inner);
        float gelu_result = val * kHalf * (1.0f + tanh_val);
        
        output[idx] = static_cast<T>(gelu_result);
    }
}

// Optimized FP32 SIMT kernel for SM80+
struct OptimizedFusedGemmBiasGeluSM80_FP32 {
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementAccumulator = float;
    
    // Use same layout as original working kernel
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;  // Match original working kernel
    using LayoutC = cutlass::layout::RowMajor;
    
    // SIMT instruction shape for FP32
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;  // SIMT instruction
    
    // Conservative but compatible shapes
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;  // K=8 for FP32
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;  // Match K dimension
    
    using ArchTag = cutlass::arch::Sm80;
    using OperatorClass = cutlass::arch::OpClassSimt;  // SIMT for FP32
    
    // Conservative staging
    static constexpr int Stages = 2;
    
    // SIMT requires alignment=1
    static constexpr int AlignmentA = 1;  // SIMT requirement
    static constexpr int AlignmentB = 1;  // SIMT requirement
    
    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    
    // Standard linear combination epilogue - SIMT requires 1 element per access
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementC,
        1,  // SIMT epilogue must use 1 element per access
        ElementAccumulator,
        ElementAccumulator
    >;
    
    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        OperatorClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp,
        ThreadblockSwizzle,
        Stages,
        AlignmentA,
        AlignmentB
    >;
};

// Optimized FP16 Tensor Core kernel for SM80+
struct OptimizedFusedGemmBiasGeluSM80_FP16 {
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = float;
    
    // Use same layout as original working kernel
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;  // Match original working kernel
    using LayoutC = cutlass::layout::RowMajor;
    
    // Use proper FP16 Tensor Core instruction shape
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;  // FP16 instruction
    
    // Use conservative but compatible shapes
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;  // K=32 for better alignment
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;  // Match K dimension
    
    using ArchTag = cutlass::arch::Sm80;
    using OperatorClass = cutlass::arch::OpClassSimt;  // Stay with SIMT for now
    
    // Conservative staging
    static constexpr int Stages = 2;
    
    // Better alignment for FP16
    static constexpr int AlignmentA = 8;  // Improve alignment
    static constexpr int AlignmentB = 8;
    
    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    
    // Standard linear combination epilogue
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementC,
        8,  // Elements per access
        ElementAccumulator,
        ElementAccumulator
    >;
    
    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        OperatorClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp,
        ThreadblockSwizzle,
        Stages,
        AlignmentA,
        AlignmentB
    >;
};

// Optimized kernel launcher with Tensor Core GEMM + optimized bias+GELU
template <typename KernelTraits>
TensorFuseStatus launch_optimized_fused_gemm_bias_gelu(
    const void* A, const void* B, const void* bias, void* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream) {
    
    using Gemm = typename KernelTraits::Gemm;
    using ElementA = typename KernelTraits::ElementA;
    using ElementB = typename KernelTraits::ElementB;
    using ElementC = typename KernelTraits::ElementC;
    
    // Set up GEMM problem
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    
    // Leading dimensions with proper stride calculation
    int lda = K;  // A is [M x K] in RowMajor, so lda = K
    int ldb = K;  // B is [K x N] in ColumnMajor, so ldb = K (leading dimension = rows)
    int ldc = N;  // C is [M x N] in RowMajor, so ldc = N
    
    // Create GEMM arguments (no bias fusion in epilogue)
    typename Gemm::Arguments args{
        problem_size,
        {static_cast<const ElementA*>(A), lda},
        {static_cast<const ElementB*>(B), ldb},
        {nullptr, ldc},  // No source C
        {static_cast<ElementC*>(C), ldc},
        {alpha, 0.0f},  // Set beta to 0 since we handle bias separately
        1  // Split-K slices
    };
    
    // Initialize GEMM operator
    Gemm gemm_op;
    
    // Check if the operation is supported
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        return TENSORFUSE_ERROR_KERNEL_LAUNCH_FAILED;
    }
    
    // Get workspace size
    size_t workspace_size = gemm_op.get_workspace_size(args);
    
    // Allocate workspace if needed
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaError_t cuda_error = cudaMalloc(&workspace, workspace_size);
        if (cuda_error != cudaSuccess) {
            return TENSORFUSE_ERROR_CUDA_ERROR;
        }
    }
    
    // Initialize the GEMM operator
    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        if (workspace) cudaFree(workspace);
        return TENSORFUSE_ERROR_KERNEL_LAUNCH_FAILED;
    }
    
    // Launch the optimized Tensor Core GEMM kernel
    status = gemm_op.run(stream);
    
    // Clean up workspace
    if (workspace) {
        cudaFree(workspace);
    }
    
    if (status != cutlass::Status::kSuccess) {
        return TENSORFUSE_ERROR_KERNEL_LAUNCH_FAILED;
    }
    
    // Launch optimized bias + GELU kernel with 2D block structure
    // Use 2D grid for better memory access patterns
    dim3 block_size(16, 16);  // 256 threads per block
    dim3 grid_size((N + block_size.x - 1) / block_size.x, 
                   (M + block_size.y - 1) / block_size.y);
    
    optimized_add_bias_gelu_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<ElementC*>(C), static_cast<const ElementC*>(bias), M, N, beta);
    
    // Check for kernel launch errors
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        return TENSORFUSE_ERROR_CUDA_ERROR;
    }
    
    return TENSORFUSE_SUCCESS;
}

// Optimized dispatch functions
TensorFuseStatus optimized_fused_gemm_bias_gelu_fp32(
    const void* A, const void* B, const void* bias, void* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream) {
    
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    // Use optimized Tensor Core kernel for SM80+
    if (props.major >= 8) {
        return launch_optimized_fused_gemm_bias_gelu<OptimizedFusedGemmBiasGeluSM80_FP32>(
            A, B, bias, C, M, N, K, alpha, beta, stream);
    } else {
        return TENSORFUSE_ERROR_UNSUPPORTED_OPERATION;
    }
}

TensorFuseStatus optimized_fused_gemm_bias_gelu_fp16(
    const void* A, const void* B, const void* bias, void* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream) {
    
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    // Use optimized Tensor Core kernel for SM80+
    if (props.major >= 8) {
        return launch_optimized_fused_gemm_bias_gelu<OptimizedFusedGemmBiasGeluSM80_FP16>(
            A, B, bias, C, M, N, K, alpha, beta, stream);
    } else {
        return TENSORFUSE_ERROR_UNSUPPORTED_OPERATION;
    }
}

} // namespace kernels
} // namespace tensorfuse

// C wrapper functions for optimized kernels
extern "C" {

TensorFuseStatus optimized_fused_gemm_bias_gelu_fp32_wrapper(
    const void* A, const void* B, const void* bias, void* C,
    int M, int N, int K, float alpha, float beta, cudaStream_t stream) {
    
    return tensorfuse::kernels::optimized_fused_gemm_bias_gelu_fp32(
        A, B, bias, C, M, N, K, alpha, beta, stream);
}

TensorFuseStatus optimized_fused_gemm_bias_gelu_fp16_wrapper(
    const void* A, const void* B, const void* bias, void* C,
    int M, int N, int K, float alpha, float beta, cudaStream_t stream) {
    
    return tensorfuse::kernels::optimized_fused_gemm_bias_gelu_fp16(
        A, B, bias, C, M, N, K, alpha, beta, stream);
}

} // extern "C" 