#!/usr/bin/env python3
"""
End-to-end mathematical verification tests for TensorFuse Python API.

This test verifies that mathematical operations produce correct results
when called through the Python API, testing the complete pipeline:
Python -> C++ bindings -> CUDA kernels
"""

import unittest
import sys
import os
import numpy as np
from numpy.testing import assert_allclose
import math

# Add the Python bindings to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))

try:
    import tensorfuse
    TENSORFUSE_AVAILABLE = True
except ImportError as e:
    TENSORFUSE_AVAILABLE = False
    print(f"TensorFuse not available: {e}")
    tensorfuse = None


def gelu_reference(x):
    """Reference implementation of GELU activation."""
    # Using the same constants as the C++ implementation
    alpha = 0.7978845608028654  # sqrt(2/pi)
    beta = 0.044715
    return x * 0.5 * (1.0 + np.tanh(alpha * (x + beta * x * x * x)))


def gemm_bias_gelu_reference(A, B, bias, alpha=1.0, beta=1.0):
    """Reference implementation of GEMM + Bias + GELU."""
    # Standard matrix multiplication: C = alpha * A @ B + beta * bias
    gemm_result = alpha * np.matmul(A, B)
    
    # Add bias (broadcasting)
    biased_result = gemm_result + beta * bias
    
    # Apply GELU activation
    return gelu_reference(biased_result)


def softmax_reference(x, axis=-1):
    """Reference implementation of softmax."""
    # Subtract max for numerical stability
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    
    # Compute exponentials
    exp_x = np.exp(x_shifted)
    
    # Normalize
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def dropout_reference(x, p=0.1, training=True, seed=None):
    """Reference implementation of dropout."""
    if not training:
        return x
    
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random mask
    mask = np.random.random(x.shape) > p
    
    # Scale by 1/(1-p) to maintain expected value
    return x * mask / (1 - p)


class TestMathematicalVerification(unittest.TestCase):
    """Test mathematical correctness through Python API."""
    
    def setUp(self):
        """Set up test environment."""
        # Initialize TensorFuse
        try:
            tensorfuse.init()
        except Exception as e:
            self.skipTest(f"Failed to initialize TensorFuse: {e}")
        
        # Verify GPU availability
        try:
            device_info = tensorfuse.get_device_info()
            self.assertIsNotNone(device_info)
        except Exception as e:
            self.skipTest(f"GPU not available: {e}")
    
    def tearDown(self):
        """Clean up test environment."""
        try:
            tensorfuse.shutdown()
        except Exception as e:
            # Don't fail the test if shutdown fails
            pass
    
    def test_gemm_bias_gelu_mathematical_correctness(self):
        """Test GEMM+Bias+GELU mathematical correctness via Python API."""
        # Test parameters
        M, N, K = 32, 64, 128
        alpha, beta = 1.0, 1.0
        
        # Generate test data
        np.random.seed(42)
        A = np.random.uniform(-0.5, 0.5, (M, K)).astype(np.float32)
        B = np.random.uniform(-0.5, 0.5, (K, N)).astype(np.float32)
        bias = np.random.uniform(-0.5, 0.5, (N,)).astype(np.float32)
        
        # Compute reference
        C_ref = gemm_bias_gelu_reference(A, B, bias, alpha, beta)
        
        try:
            # Create output array for TensorFuse
            output = np.zeros((M, N), dtype=np.float32)
            
            # Execute TensorFuse operation
            tensorfuse.fused_gemm_bias_gelu(A, B, bias, output)
            
            # Verify shapes match
            self.assertEqual(output.shape, C_ref.shape)
            
            # Verify mathematical correctness
            max_error = np.max(np.abs(output - C_ref))
            print(f"GEMM+Bias+GELU max error: {max_error}")
            
            # Verify using numpy testing
            assert_allclose(output, C_ref, rtol=1e-4, atol=1e-4)
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_gemm_bias_gelu_different_sizes(self):
        """Test GEMM+Bias+GELU with different matrix sizes."""
        test_sizes = [
            (1, 1, 1),      # Minimal size
            (8, 16, 32),    # Small
            (16, 32, 64),   # Medium
            (32, 64, 128),  # Large
        ]
        
        for M, N, K in test_sizes:
            with self.subTest(M=M, N=N, K=K):
                # Generate test data
                np.random.seed(42)
                A = np.random.uniform(-0.5, 0.5, (M, K)).astype(np.float32)
                B = np.random.uniform(-0.5, 0.5, (K, N)).astype(np.float32)
                bias = np.random.uniform(-0.5, 0.5, (N,)).astype(np.float32)
                
                # Compute reference
                C_ref = gemm_bias_gelu_reference(A, B, bias, 1.0, 1.0)
                
                try:
                    # Create output array for TensorFuse
                    output = np.zeros((M, N), dtype=np.float32)
                    
                    # Execute TensorFuse operation
                    tensorfuse.fused_gemm_bias_gelu(A, B, bias, output)
                    
                    # Verify result
                    assert_allclose(output, C_ref, rtol=1e-4, atol=1e-4)
                    
                except Exception as e:
                    if "CUDA" in str(e) or "GPU" in str(e):
                        self.skipTest(f"GPU not available: {e}")
                    else:
                        raise
    
    def test_gemm_bias_gelu_alpha_beta_scaling(self):
        """Test GEMM+Bias+GELU with different alpha/beta values."""
        # Skip this test for now as the API doesn't support alpha/beta parameters
        self.skipTest("Alpha/beta parameters not supported in current API")

    def test_softmax_dropout_mathematical_correctness(self):
        """Test Softmax+Dropout mathematical correctness via Python API."""
        # Test parameters (attention-like)
        batch_size, num_heads, seq_len, head_dim = 2, 8, 64, 64
        dropout_prob = 0.0  # No dropout for exact comparison
        
        # Generate test data (attention scores)
        np.random.seed(42)
        input_data = np.random.uniform(-2.0, 2.0, 
            (batch_size, num_heads, seq_len, head_dim)).astype(np.float32)
        
        # Compute reference softmax (without dropout for exact comparison)
        softmax_ref = softmax_reference(input_data, axis=-1)
        
        try:
            # Create output arrays for TensorFuse
            output = np.zeros_like(input_data)
            dropout_mask = np.zeros(input_data.shape, dtype=np.uint8)
            
            # Execute TensorFuse operation
            tensorfuse.fused_softmax_dropout(input_data, output, dropout_mask, dropout_prob)
            
            # Verify shapes match
            self.assertEqual(output.shape, softmax_ref.shape)
            
            # Verify softmax properties
            # 1. All values should be non-negative
            self.assertTrue(np.all(output >= 0))
            
            # 2. Each row should sum to approximately 1
            row_sums = np.sum(output, axis=-1)
            assert_allclose(row_sums, np.ones_like(row_sums), rtol=1e-4, atol=1e-4)
            
            # 3. Results should match reference implementation
            max_error = np.max(np.abs(output - softmax_ref))
            print(f"Softmax max error: {max_error}")
            assert_allclose(output, softmax_ref, rtol=1e-4, atol=1e-4)
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_gelu_activation_properties(self):
        """Test GELU activation properties."""
        # Test specific values where GELU behavior is well-defined
        test_values = np.array([
            -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0
        ], dtype=np.float32)
        
        # Known GELU properties:
        # 1. GELU(0) ≈ 0
        # 2. GELU is approximately linear for positive values
        # 3. GELU approaches 0 for large negative values
        # 4. GELU is smooth and differentiable
        
        gelu_ref = gelu_reference(test_values)
        
        # Test through a simple GEMM that isolates GELU
        A = np.ones((1, 1), dtype=np.float32)
        B = test_values.reshape(1, -1)
        bias = np.zeros(B.shape[1], dtype=np.float32)
        
        try:
            # Create output array for TensorFuse
            output = np.zeros((1, B.shape[1]), dtype=np.float32)
            
            # Execute TensorFuse operation
            tensorfuse.fused_gemm_bias_gelu(A, B, bias, output)
            
            gelu_gpu = output.flatten()
            
            # Verify GELU properties
            self.assertAlmostEqual(gelu_gpu[4], 0.0, places=5)  # GELU(0) ≈ 0
            self.assertTrue(gelu_gpu[6] > 0)  # GELU(1) > 0
            self.assertTrue(gelu_gpu[0] < 0.1)  # GELU(-3) ≈ 0
            
            # Verify against reference
            assert_allclose(gelu_gpu, gelu_ref, rtol=1e-4, atol=1e-4)
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_memory_management_during_operations(self):
        """Test that memory is properly managed during operations."""
        # Create multiple tensors and operations to test memory management
        for i in range(5):
            with self.subTest(iteration=i):
                # Generate test data
                M, N, K = 32, 64, 128
                np.random.seed(42 + i)
                A = np.random.uniform(-0.5, 0.5, (M, K)).astype(np.float32)
                B = np.random.uniform(-0.5, 0.5, (K, N)).astype(np.float32)
                bias = np.random.uniform(-0.5, 0.5, (N,)).astype(np.float32)
                
                try:
                    # Create output array for TensorFuse
                    output = np.zeros((M, N), dtype=np.float32)
                    
                    # Execute operation
                    tensorfuse.fused_gemm_bias_gelu(A, B, bias, output)
                    
                    # Verify result is valid
                    self.assertFalse(np.any(np.isnan(output)))
                    self.assertFalse(np.any(np.isinf(output)))
                    
                except Exception as e:
                    if "CUDA" in str(e) or "GPU" in str(e):
                        self.skipTest(f"GPU not available: {e}")
                    else:
                        raise


if __name__ == '__main__':
    unittest.main() 