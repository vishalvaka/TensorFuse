#!/usr/bin/env python3
"""Debug script to check scale value passing to INT8 kernel"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))

import numpy as np
import tensorfuse

def debug_scale_passing():
    print("🔍 Debug Scale Passing to INT8 Kernel")
    print("=" * 50)
    
    # Initialize TensorFuse
    tensorfuse.init()
    
    # Create simple test data
    M, N, K = 16, 16, 16  # Small matrices for easier debugging
    
    # Create FP32 reference data
    np.random.seed(42)
    A_fp32 = np.random.uniform(-0.5, 0.5, (M, K)).astype(np.float32)
    B_fp32 = np.random.uniform(-0.5, 0.5, (K, N)).astype(np.float32)
    bias_fp32 = np.zeros(N, dtype=np.float32)  # Use zero bias for simplicity
    
    # Quantize to INT8
    scale_A = np.max(np.abs(A_fp32)) / 127.0
    scale_B = np.max(np.abs(B_fp32)) / 127.0
    
    A_int8 = np.round(A_fp32 / scale_A).astype(np.int8)
    B_int8 = np.round(B_fp32 / scale_B).astype(np.int8)
    
    print(f"Scale A: {scale_A:.6f}")
    print(f"Scale B: {scale_B:.6f}")
    print(f"Combined scale: {scale_A * scale_B:.6f}")
    print(f"Expected dequant scale: {1.0 / (scale_A * scale_B):.6f}")
    
    # Compute reference
    ref_gemm = A_fp32 @ B_fp32
    ref_output = np.maximum(0, ref_gemm + bias_fp32)  # Simple ReLU instead of GELU for debugging
    
    print(f"Reference GEMM range: [{ref_gemm.min():.6f}, {ref_gemm.max():.6f}]")
    print(f"Reference output range: [{ref_output.min():.6f}, {ref_output.max():.6f}]")
    
    # Test INT8 kernel
    output_int8 = np.zeros((M, N), dtype=np.float32)
    
    try:
        tensorfuse.fused_gemm_bias_gelu(A_int8, B_int8, bias_fp32, output_int8)
        print(f"INT8 output range: [{output_int8.min():.6f}, {output_int8.max():.6f}]")
        
        # Check if output is reasonable
        if np.allclose(output_int8, 0.0):
            print("⚠️  INT8 output is all zeros!")
        elif np.any(output_int8 > 100):
            print("⚠️  INT8 output has very large values!")
        elif np.any(output_int8 < -100):
            print("⚠️  INT8 output has very small values!")
        else:
            print("✅ INT8 output seems reasonable")
            
        # Compare specific values
        print(f"Reference[0,0]: {ref_output[0,0]:.6f}")
        print(f"INT8[0,0]: {output_int8[0,0]:.6f}")
        print(f"Ratio: {output_int8[0,0] / ref_output[0,0] if ref_output[0,0] != 0 else 'N/A'}")
        
    except Exception as e:
        print(f"❌ INT8 kernel failed: {e}")

if __name__ == "__main__":
    debug_scale_passing() 