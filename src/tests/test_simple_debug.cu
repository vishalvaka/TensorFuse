/*
 * Simple debug test for CUTLASS kernel issues
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/epilogue/thread/linear_combination.h>

// Test basic CUTLASS GEMM configuration
struct SimpleGemmFP32 {
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementAccumulator = float;
    
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    
    using ArchTag = cutlass::arch::Sm80;
    using OperatorClass = cutlass::arch::OpClassSimt;
    
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementC, 1, ElementAccumulator, ElementAccumulator>;
    
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
        EpilogueOp
    >;
};

int main() {
    std::cout << "CUTLASS Debug Test" << std::endl;
    
    // Small problem size
    int M = 16, N = 32, K = 64;
    
    // Test basic CUTLASS configuration
    using Gemm = SimpleGemmFP32::Gemm;
    
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    
    typename Gemm::Arguments args{
        problem_size,
        {nullptr, K},
        {nullptr, K},
        {nullptr, N},
        {nullptr, N},
        {1.0f, 0.0f}
    };
    
    Gemm gemm_op;
    
    // Check if the operation is supported
    cutlass::Status status = gemm_op.can_implement(args);
    
    std::cout << "can_implement status: " << int(status) << std::endl;
    
    if (status == cutlass::Status::kSuccess) {
        std::cout << "CUTLASS kernel configuration is supported!" << std::endl;
    } else {
        std::cout << "CUTLASS kernel configuration is NOT supported!" << std::endl;
        std::cout << "Status: " << int(status) << std::endl;
    }
    
    return 0;
} 