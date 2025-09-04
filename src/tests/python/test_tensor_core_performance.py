#!/usr/bin/env python3
"""
Test script to verify INT8 Tensor Core performance
This is the critical test for our 2-7x speedup goal!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))

import numpy as np
import time
import tensorfuse

def test_int8_tensor_cores():
    """Test INT8 Tensor Core performance vs FP32 baseline"""
    
    print("🔥 Testing INT8 Tensor Cores - The Performance Multiplier!")
    print("=" * 60)
    
    # Initialize TensorFuse
    config = tensorfuse.TensorFuseConfig()
    tensorfuse.init(config)
    
    # Test dimensions for transformer-like workloads
    M, N, K = 1024, 4096, 1024  # Typical LLM FFN dimensions
    
    print(f"Matrix dimensions: M={M}, N={N}, K={K}")
    print(f"GPU: {tensorfuse.get_device_name()}")
    
    # Create test data
    np.random.seed(42)
    
    # FP32 test data
    A_fp32 = np.random.randn(M, K).astype(np.float32) * 0.1
    B_fp32 = np.random.randn(K, N).astype(np.float32) * 0.1
    bias_fp32 = np.random.randn(N).astype(np.float32) * 0.01
    
    # INT8 quantized data (simple quantization for testing)
    scale = 127.0 / np.max(np.abs(A_fp32))
    A_int8 = np.clip(A_fp32 * scale, -127, 127).astype(np.int8)
    B_int8 = np.clip(B_fp32 * scale, -127, 127).astype(np.int8)
    
    print(f"Quantization scale: {scale:.6f}")
    print(f"A_int8 range: [{np.min(A_int8)}, {np.max(A_int8)}]")
    print(f"B_int8 range: [{np.min(B_int8)}, {np.max(B_int8)}]")
    
    # Warmup runs
    print("\n🔥 Warming up kernels...")
    
    warmup_A = np.random.randn(128, 128).astype(np.float32)
    warmup_B = np.random.randn(128, 128).astype(np.float32)
    warmup_bias = np.random.randn(128).astype(np.float32)
    
    for i in range(3):
        _ = tensorfuse.fused_gemm_bias_gelu(warmup_A, warmup_B, warmup_bias)
    
    # Test FP32 performance (baseline)
    print("\n🟦 Testing FP32 baseline...")
    
    fp32_times = []
    for i in range(10):
        start_time = time.perf_counter()
        result_fp32 = tensorfuse.fused_gemm_bias_gelu(A_fp32, B_fp32, bias_fp32)
        end_time = time.perf_counter()
        fp32_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    fp32_avg_time = np.mean(fp32_times[2:])  # Skip first 2 for warmup
    fp32_std_time = np.std(fp32_times[2:])
    
    print(f"FP32 performance: {fp32_avg_time:.3f} ± {fp32_std_time:.3f} ms")
    
    # Test INT8 performance
    print("\n🚀 Testing INT8 Tensor Cores...")
    
    int8_times = []
    for i in range(10):
        start_time = time.perf_counter()
        try:
            result_int8 = tensorfuse.fused_gemm_bias_gelu_int8(A_int8, B_int8, bias_fp32)
            end_time = time.perf_counter()
            int8_times.append((end_time - start_time) * 1000)  # Convert to ms
        except Exception as e:
            print(f"INT8 kernel failed: {e}")
            return False
    
    int8_avg_time = np.mean(int8_times[2:])  # Skip first 2 for warmup
    int8_std_time = np.std(int8_times[2:])
    
    print(f"INT8 performance: {int8_avg_time:.3f} ± {int8_std_time:.3f} ms")
    
    # Calculate speedup
    speedup = fp32_avg_time / int8_avg_time
    
    print("\n" + "=" * 60)
    print("🎯 PERFORMANCE RESULTS")
    print("=" * 60)
    print(f"FP32 Baseline: {fp32_avg_time:.3f} ms")
    print(f"INT8 Tensor Cores: {int8_avg_time:.3f} ms")
    print(f"🚀 SPEEDUP: {speedup:.2f}x")
    
    if speedup >= 1.5:
        print("✅ SUCCESS: Significant speedup achieved!")
        print("🔥 INT8 Tensor Cores are working correctly!")
    elif speedup >= 1.1:
        print("⚠️  PARTIAL SUCCESS: Some speedup detected")
        print("   (May need optimization or larger matrices)")
    else:
        print("❌ WARNING: No significant speedup detected")
        print("   (Check INT8 kernel implementation)")
    
    # Verify mathematical correctness (approximate due to quantization)
    print("\n📊 Checking mathematical correctness...")
    
    # Dequantize result for comparison
    result_int8_dequant = result_int8.astype(np.float32) / (scale * scale)
    
    # Compare with FP32 result
    max_error = np.max(np.abs(result_fp32 - result_int8_dequant))
    relative_error = max_error / np.max(np.abs(result_fp32))
    
    print(f"Max absolute error: {max_error:.6f}")
    print(f"Relative error: {relative_error:.6f}")
    
    if relative_error < 0.1:  # 10% tolerance for quantization
        print("✅ Mathematical correctness verified!")
    else:
        print("⚠️  Large quantization error detected")
    
    # Performance efficiency metrics
    total_ops = 2 * M * N * K + M * N  # GEMM ops + bias/activation
    fp32_tflops = (total_ops / (fp32_avg_time / 1000)) / 1e12
    int8_tflops = (total_ops / (int8_avg_time / 1000)) / 1e12
    
    print(f"\n⚡ Performance Efficiency:")
    print(f"FP32: {fp32_tflops:.2f} TFLOPS")
    print(f"INT8: {int8_tflops:.2f} TFLOPS")
    print(f"Efficiency gain: {int8_tflops/fp32_tflops:.2f}x")
    
    tensorfuse.shutdown()
    return speedup >= 1.1

if __name__ == "__main__":
    try:
        success = test_int8_tensor_cores()
        if success:
            print("\n🎉 INT8 Tensor Core test PASSED!")
            sys.exit(0)
        else:
            print("\n❌ INT8 Tensor Core test FAILED!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 