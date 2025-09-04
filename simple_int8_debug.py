#!/usr/bin/env python3
"""Simple debug script to understand INT8 kernel behavior"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))

import numpy as np
import tensorfuse

def simple_int8_debug():
    print("🔍 Simple INT8 Debug")
    print("=" * 40)
    
    # Initialize TensorFuse
    tensorfuse.init()
    
    # Use the exact same setup as the test
    M, N, K = 64, 128, 64
    
    # Create FP32 reference data (same as test)
    np.random.seed(123)
    A_fp32 = np.random.uniform(-1.0, 1.0, (M, K)).astype(np.float32)
    B_fp32 = np.random.uniform(-1.0, 1.0, (K, N)).astype(np.float32)
    bias_fp32 = np.random.uniform(-0.1, 0.1, (N,)).astype(np.float32)
    
    # Quantize to INT8 (same as test)
    scale_A = 127.0 / np.max(np.abs(A_fp32))
    scale_B = 127.0 / np.max(np.abs(B_fp32))
    
    A_int8 = np.clip(A_fp32 * scale_A, -127, 127).astype(np.int8)
    B_int8 = np.clip(B_fp32 * scale_B, -127, 127).astype(np.int8)
    
    print(f"Scales: A={scale_A:.3f}, B={scale_B:.3f}")
    print(f"Combined scale: {scale_A * scale_B:.6f}")
    
    # Manual GELU implementation (same as test)
    def gelu(x):
        return x * 0.5 * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    
    # Reference computation
    gemm_result = np.matmul(A_fp32, B_fp32) + bias_fp32
    reference = gelu(gemm_result)
    
    print(f"Reference gemm_result range: [{gemm_result.min():.6f}, {gemm_result.max():.6f}]")
    print(f"Reference output range: [{reference.min():.6f}, {reference.max():.6f}]")
    
    # INT8 computation
    output_int8 = np.zeros((M, N), dtype=np.float32)
    tensorfuse.fused_gemm_bias_gelu(A_int8, B_int8, bias_fp32, output_int8)
    
    print(f"INT8 raw output range: [{output_int8.min():.6f}, {output_int8.max():.6f}]")
    
    # Try different dequantization approaches
    print("\nTrying different dequantization:")
    
    # Approach 1: Test's approach
    dequant1 = output_int8 / (scale_A * scale_B)
    print(f"1. output_int8 / (scale_A * scale_B): [{dequant1.min():.6f}, {dequant1.max():.6f}]")
    
    # Approach 2: No dequantization
    print(f"2. output_int8 (raw): [{output_int8.min():.6f}, {output_int8.max():.6f}]")
    
    # Compare a few sample values
    print(f"\nSample comparison (first 3 values):")
    print(f"Reference:  {reference[0, :3]}")
    print(f"Dequant1:   {dequant1[0, :3]}")
    print(f"Raw INT8:   {output_int8[0, :3]}")
    
    # Cleanup
    tensorfuse.shutdown()

if __name__ == "__main__":
    simple_int8_debug() 