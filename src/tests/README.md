# TensorFuse Tests

This directory contains various test files for the TensorFuse library.

## Test Files

### `test_working_kernel.cu`
**Purpose**: Main functionality test for the fused GEMM+Bias+GELU kernels  
**Status**: ✅ Working  
**Description**: Tests both FP32 and FP16 kernels with simple input data and verifies GPU vs CPU results match exactly.

### `test_kernels.cu`
**Purpose**: Comprehensive test with random data and detailed error analysis  
**Status**: ⚠️ In development  
**Description**: More thorough testing with random input data and detailed numerical accuracy validation.

### `test_kernels_simple.cpp`
**Purpose**: Basic library loading test  
**Status**: ✅ Working  
**Description**: Simple test that verifies the TensorFuse library can be loaded and kernel functions can be found.

### `test_simple_debug.cu`
**Purpose**: CUTLASS configuration debugging  
**Status**: ✅ Working  
**Description**: Debug utility to test basic CUTLASS GEMM configuration without TensorFuse integration.

## Running Tests

From the `src/tests` directory:

```bash
# Compile and run the main working test
nvcc -o test_working_kernel test_working_kernel.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -ldl -std=c++17 --expt-relaxed-constexpr
./test_working_kernel

# Compile and run the library loading test
g++ -o test_kernels_simple test_kernels_simple.cpp -ldl -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart
./test_kernels_simple

# Compile and run the CUTLASS debug test
nvcc -o test_simple_debug test_simple_debug.cu -I/usr/local/cuda/include -I../../src -L/usr/local/cuda/lib64 -lcudart -std=c++17 --expt-relaxed-constexpr
./test_simple_debug
```

## Test Results

Current status: **All core functionality tests PASSING** ✅

- Library builds successfully
- Kernels execute without errors  
- GPU vs CPU results match exactly
- Memory management working correctly
- CUTLASS integration functional 