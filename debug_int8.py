#!/usr/bin/env python3
"""Debug script for INT8 Tensor Core issues"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))

import numpy as np
import tensorfuse

def debug_int8():
    print("🔍 Debugging INT8 Tensor Core Implementation")
    print("=" * 50)
    
    # Initialize TensorFuse
    try:
        tensorfuse.init()
        print("✅ TensorFuse initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize TensorFuse: {e}")
        return
    
    # Test 1: Basic functionality check
    print("\n1. Testing basic INT8 functionality:")
    
    # Use larger test matrices that meet INT8 Tensor Core requirements
    M, N, K = 64, 128, 64  # Larger matrices
    
    # Create INT8 test data
    np.random.seed(42)
    A_int8 = np.random.randint(-50, 50, (M, K), dtype=np.int8)
    B_int8 = np.random.randint(-50, 50, (K, N), dtype=np.int8)
    bias_fp32 = np.random.randn(N).astype(np.float32) * 0.1
    
    print(f"   A_int8 shape: {A_int8.shape}, range: [{A_int8.min()}, {A_int8.max()}]")
    print(f"   B_int8 shape: {B_int8.shape}, range: [{B_int8.min()}, {B_int8.max()}]")
    print(f"   bias_fp32 shape: {bias_fp32.shape}, range: [{bias_fp32.min():.4f}, {bias_fp32.max():.4f}]")
    
    # Create output tensor
    output = np.zeros((M, N), dtype=np.float32)
    
    # Test the operation
    try:
        tensorfuse.fused_gemm_bias_gelu(A_int8, B_int8, bias_fp32, output)
        print(f"✅ INT8 operation completed")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.6f}, {output.max():.6f}]")
        print(f"   Contains NaN: {np.any(np.isnan(output))}")
        print(f"   Contains Inf: {np.any(np.isinf(output))}")
        print(f"   All zeros: {np.all(output == 0)}")
        print(f"   Sample values: {output[0, :5]}")
        
        if np.any(np.isnan(output)):
            print("❌ Output contains NaN values!")
        elif np.any(np.isinf(output)):
            print("❌ Output contains Inf values!")
        elif np.all(output == 0):
            print("⚠️  Output is all zeros")
        else:
            print("✅ Output looks reasonable")
            
    except Exception as e:
        print(f"❌ INT8 operation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Compare with FP32 baseline
    print("\n2. Comparing with FP32 baseline:")
    
    # Create equivalent FP32 data
    A_fp32 = A_int8.astype(np.float32)
    B_fp32 = B_int8.astype(np.float32)
    output_fp32 = np.zeros((M, N), dtype=np.float32)
    
    try:
        tensorfuse.fused_gemm_bias_gelu(A_fp32, B_fp32, bias_fp32, output_fp32)
        print(f"✅ FP32 operation completed")
        print(f"   FP32 output range: [{output_fp32.min():.6f}, {output_fp32.max():.6f}]")
        print(f"   FP32 sample values: {output_fp32[0, :5]}")
        
        if not np.any(np.isnan(output)) and not np.any(np.isnan(output_fp32)):
            diff = np.abs(output - output_fp32)
            print(f"   Max difference: {diff.max():.6f}")
            print(f"   Mean difference: {diff.mean():.6f}")
        
    except Exception as e:
        print(f"❌ FP32 operation failed: {e}")
    
    # Test 3: Test with different data ranges
    print("\n3. Testing with different data ranges:")
    
    test_ranges = [
        (-10, 10),   # Small range
        (-127, 127), # Full INT8 range
        (-5, 5),     # Very small range
    ]
    
    for i, (low, high) in enumerate(test_ranges):
        print(f"\n   Test {i+1}: Range [{low}, {high}]")
        
        A_test = np.random.randint(low, high, (M, K), dtype=np.int8)
        B_test = np.random.randint(low, high, (K, N), dtype=np.int8)
        bias_test = np.random.uniform(-0.01, 0.01, N).astype(np.float32)
        output_test = np.zeros((M, N), dtype=np.float32)
        
        try:
            tensorfuse.fused_gemm_bias_gelu(A_test, B_test, bias_test, output_test)
            print(f"      ✅ Range [{low}, {high}] - Output: [{output_test.min():.6f}, {output_test.max():.6f}]")
            print(f"         NaN: {np.any(np.isnan(output_test))}, Inf: {np.any(np.isinf(output_test))}")
        except Exception as e:
            print(f"      ❌ Range [{low}, {high}] failed: {e}")
    
    # Cleanup
    try:
        tensorfuse.shutdown()
        print("\n✅ TensorFuse shutdown successfully")
    except Exception as e:
        print(f"❌ Failed to shutdown TensorFuse: {e}")

if __name__ == "__main__":
    debug_int8() 