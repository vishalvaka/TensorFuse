/*
 * Copyright (c) 2024, TensorFuse Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Fused GEMM+Bias+GELU kernel implementation using CUTLASS
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

// Forward declarations
TensorFuseStatus fused_gemm_bias_gelu_fp32(
    const void* A, const void* B, const void* bias, void* C,
    int M, int N, int K, float alpha, float beta, cudaStream_t stream);

// GELU activation function for epilogue
template <typename T>
struct GeluActivation {
    CUTLASS_HOST_DEVICE
    T operator()(T const& x) const {
        // Approximate GELU: x * 0.5 * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        T const kAlpha = T(0.7978845608028654);  // sqrt(2/pi)
        T const kBeta = T(0.044715);
        T const kHalf = T(0.5);
        T const kOne = T(1.0);
        
        T x_cubed = x * x * x;
        T inner = kAlpha * (x + kBeta * x_cubed);
        T tanh_inner = cutlass::fast_tanh(inner);
        return x * kHalf * (kOne + tanh_inner);
    }
};

// FP32 SIMT kernel for SM80 (Ampere)
struct FusedGemmBiasGeluKernelSM80_FP32 {
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementAccumulator = float;
    
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;  // Fixed: Both matrices are row-major
    using LayoutC = cutlass::layout::RowMajor;
    
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    
    using ArchTag = cutlass::arch::Sm80;
    using OperatorClass = cutlass::arch::OpClassSimt;
    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementC,
        1,
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
        2  // Stages for SIMT
    >;
};

// FP32 SIMT kernel for SM89 (Ada Lovelace) - Use SM80 config since CUTLASS doesn't have SM89 defaults
struct FusedGemmBiasGeluKernelSM89_FP32 {
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementAccumulator = float;
    
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;  // Fixed: Both matrices are row-major
    using LayoutC = cutlass::layout::RowMajor;
    
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    
    // Use SM80 arch tag since CUTLASS doesn't have SM89 default configs
    using ArchTag = cutlass::arch::Sm80;
    using OperatorClass = cutlass::arch::OpClassSimt;
    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementC,
        1,
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
        2  // Stages for SIMT
    >;
};



// FP16 Tensor Core kernel - Use SM80 for compatibility
struct FusedGemmBiasGeluKernelSM80_FP16 {
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = float;
    
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    
    // Tensor Core instruction shape for FP16
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    
    // Optimal warp shape for FP16 Tensor Cores
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    
    // Threadblock shape optimized for performance
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
    
    // Use SM80 for compatibility
    using ArchTag = cutlass::arch::Sm80;
    using OperatorClass = cutlass::arch::OpClassTensorOp;  // Use Tensor Cores!
    
    // Stages for optimal performance
    static constexpr int Stages = 3;
    
    // Alignment for FP16 operations
    static constexpr int AlignmentA = 8;
    static constexpr int AlignmentB = 8;
    
    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    
    // Simple linear combination epilogue
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

// Simple CUDA kernel to add bias and apply GELU activation
template <typename T>
__global__ void add_bias_gelu_kernel(
    T* output, const T* bias, int M, int N, float beta) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = M * N;
    
    if (idx < total_elements) {
        int col = idx % N;
        
        // Add bias
        float val = static_cast<float>(output[idx]) + beta * static_cast<float>(bias[col]);
        
        // Apply GELU activation
        const float kAlpha = 0.7978845608028654f;  // sqrt(2/pi)
        const float kBeta = 0.044715f;
        val = val * 0.5f * (1.0f + tanh(kAlpha * (val + kBeta * val * val * val)));
        
        output[idx] = static_cast<T>(val);
    }
}

// Kernel launcher template
template <typename KernelTraits>
TensorFuseStatus launch_fused_gemm_bias_gelu_kernel(
    const void* A, const void* B, const void* bias, void* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream) {
    
    using Gemm = typename KernelTraits::Gemm;
    using ElementA = typename KernelTraits::ElementA;
    using ElementB = typename KernelTraits::ElementB;
    using ElementC = typename KernelTraits::ElementC;
    
    // Set up GEMM arguments
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    
    // Leading dimensions
    int lda = K;
    int ldb = N;  // Fixed: B is [K x N] so leading dimension is N
    int ldc = N;
    int ldd = N;
    
    // Create GEMM arguments without bias first (we'll add bias in a separate step)
    typename Gemm::Arguments args{
        problem_size,
        {static_cast<const ElementA*>(A), lda},
        {static_cast<const ElementB*>(B), ldb},
        {nullptr, ldc},  // No initial C matrix (bias will be added separately)
        {static_cast<ElementC*>(C), ldd},
        {alpha, 0.0f},  // Set beta to 0 since we'll handle bias separately
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
    
    // Launch the GEMM kernel
    status = gemm_op.run(stream);
    
    // Clean up workspace
    if (workspace) {
        cudaFree(workspace);
    }
    
    if (status != cutlass::Status::kSuccess) {
        return TENSORFUSE_ERROR_KERNEL_LAUNCH_FAILED;
    }
    
    // Now add bias and apply GELU activation
    int total_elements = M * N;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    add_bias_gelu_kernel<<<blocks, threads_per_block, 0, stream>>>(
        static_cast<ElementC*>(C), static_cast<const ElementC*>(bias), M, N, beta);
    
    // Check for kernel launch errors
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        return TENSORFUSE_ERROR_CUDA_ERROR;
    }
    
    return TENSORFUSE_SUCCESS;
}

// Explicit instantiations and dispatch function
TensorFuseStatus fused_gemm_bias_gelu_fp32(
    const void* A, const void* B, const void* bias, void* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream) {
    
    // Try optimized implementation first, fallback to original if it fails
    extern TensorFuseStatus optimized_fused_gemm_bias_gelu_fp32(
        const void* A, const void* B, const void* bias, void* C,
        int M, int N, int K, float alpha, float beta, cudaStream_t stream);
    
    TensorFuseStatus status = optimized_fused_gemm_bias_gelu_fp32(A, B, bias, C, M, N, K, alpha, beta, stream);
    
    // If optimized version fails, fall back to original implementation
    if (status != TENSORFUSE_SUCCESS) {
        return launch_fused_gemm_bias_gelu_kernel<FusedGemmBiasGeluKernelSM80_FP32>(
            A, B, bias, C, M, N, K, alpha, beta, stream);
    }
    
    return status;
}

TensorFuseStatus fused_gemm_bias_gelu_fp16(
    const void* A, const void* B, const void* bias, void* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream) {
    
    // Try optimized implementation first, fallback to original if it fails
    extern TensorFuseStatus optimized_fused_gemm_bias_gelu_fp16(
        const void* A, const void* B, const void* bias, void* C,
        int M, int N, int K, float alpha, float beta, cudaStream_t stream);
    
    TensorFuseStatus status = optimized_fused_gemm_bias_gelu_fp16(A, B, bias, C, M, N, K, alpha, beta, stream);
    
    // If optimized version fails, fall back to original implementation
    if (status != TENSORFUSE_SUCCESS) {
        return launch_fused_gemm_bias_gelu_kernel<FusedGemmBiasGeluKernelSM80_FP16>(
            A, B, bias, C, M, N, K, alpha, beta, stream);
    }
    
    return status;
}

} // namespace kernels
} // namespace tensorfuse 

// C wrapper functions for C API
extern "C" {

TensorFuseStatus fused_gemm_bias_gelu_fp16_wrapper(
    const void* A, const void* B, const void* bias, void* C,
    int M, int N, int K, float alpha, float beta, cudaStream_t stream) {
    
    return tensorfuse::kernels::fused_gemm_bias_gelu_fp16(
        A, B, bias, C, M, N, K, alpha, beta, stream);
}

TensorFuseStatus fused_gemm_bias_gelu_fp32_wrapper(
    const void* A, const void* B, const void* bias, void* C,
    int M, int N, int K, float alpha, float beta, cudaStream_t stream) {
    
    return tensorfuse::kernels::fused_gemm_bias_gelu_fp32(
        A, B, bias, C, M, N, K, alpha, beta, stream);
}

} // extern "C"