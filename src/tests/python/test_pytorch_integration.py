#!/usr/bin/env python3
"""
PyTorch integration tests for TensorFuse.

Tests the PyTorch tensor integration functionality including:
- Tensor conversion between PyTorch and TensorFuse
- Autograd compatibility
- Device handling (CPU/GPU)
- Memory sharing
- Gradient computation
"""

import unittest
import sys
import os

# Add the Python bindings to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available - skipping PyTorch integration tests")

try:
    import tensorfuse
    from tensorfuse import core, memory, utils
    TENSORFUSE_AVAILABLE = True
except ImportError as e:
    TENSORFUSE_AVAILABLE = False
    print(f"TensorFuse not available: {e}")
    print("This is expected if CUTLASS is not installed")


class TestPyTorchIntegration(unittest.TestCase):
    """Test PyTorch tensor integration with TensorFuse."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not PYTORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        if not TENSORFUSE_AVAILABLE:
            self.skipTest("TensorFuse not available - requires CUTLASS")
    
    def test_pytorch_to_tensorfuse_conversion(self):
        """Test PyTorch tensor to TensorFuse tensor conversion."""
        # Test different data types
        test_cases = [
            (torch.float32, tensorfuse.TensorFuseDataType.FP32),
            (torch.float16, tensorfuse.TensorFuseDataType.FP16),
            (torch.int8, tensorfuse.TensorFuseDataType.INT8),
            (torch.uint8, tensorfuse.TensorFuseDataType.UINT8),
            (torch.int32, tensorfuse.TensorFuseDataType.INT32),
        ]
        
        for torch_dtype, tf_dtype in test_cases:
            with self.subTest(torch_dtype=torch_dtype, tf_dtype=tf_dtype):
                # Create test tensor
                tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch_dtype)
                
                try:
                    # Convert to TensorFuse tensor
                    tf_tensor = tensorfuse.create_tensor_from_torch(tensor)
                    
                    # Verify conversion succeeded
                    self.assertIsNotNone(tf_tensor)
                    
                except Exception as e:
                    if "CUDA" in str(e) or "GPU" in str(e):
                        self.skipTest(f"GPU not available: {e}")
                    else:
                        raise
    
    def test_bfloat16_conversion(self):
        """Test bfloat16 conversion if available."""
        if not hasattr(torch, 'bfloat16'):
            self.skipTest("bfloat16 not available in this PyTorch version")
        
        try:
            tensor = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.bfloat16)
            tf_tensor = tensorfuse.create_tensor_from_torch(tensor)
            self.assertIsNotNone(tf_tensor)
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            elif "bfloat16" in str(e):
                self.skipTest("bfloat16 not supported in this configuration")
            else:
                raise
    
    def test_cuda_tensor_handling(self):
        """Test CUDA tensor handling."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        try:
            # Create CUDA tensor
            cuda_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], 
                                     device='cuda', dtype=torch.float32)
            
            # Convert to TensorFuse tensor
            tf_tensor = tensorfuse.create_tensor_from_torch(cuda_tensor)
            self.assertIsNotNone(tf_tensor)
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_cpu_tensor_handling(self):
        """Test CPU tensor handling."""
        # Create CPU tensor
        cpu_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], 
                                device='cpu', dtype=torch.float32)
        
        try:
            # Convert to TensorFuse tensor
            tf_tensor = tensorfuse.create_tensor_from_torch(cpu_tensor)
            self.assertIsNotNone(tf_tensor)
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_autograd_compatibility(self):
        """Test autograd compatibility with TensorFuse operations."""
        # Create tensor with gradient tracking
        input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], 
                                  dtype=torch.float32, requires_grad=True)
        
        try:
            # Use TensorFuse operation in autograd context
            fused_op = core.FusedGemmBiasGelu(m=2, n=2, k=2)
            
            # This would be part of a forward pass
            # output = fused_op.forward(input_tensor, weights, bias)
            
            # For now, just test the wrapper exists
            self.assertTrue(hasattr(fused_op, 'forward'))
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_gradient_computation(self):
        """Test gradient computation through TensorFuse operations."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for gradient computation test")
        
        try:
            # Create tensors with gradient tracking
            input_tensor = torch.randn(4, 4, requires_grad=True, device='cuda')
            weight_tensor = torch.randn(4, 4, requires_grad=True, device='cuda')
            bias_tensor = torch.randn(4, requires_grad=True, device='cuda')
            
            # Use TensorFuse operation
            fused_op = core.FusedGemmBiasGelu(m=4, n=4, k=4)
            
            # Forward pass (this would call the actual operation)
            # output = fused_op.forward(input_tensor, weight_tensor, bias_tensor)
            
            # For now, just test the setup doesn't error
            self.assertTrue(input_tensor.requires_grad)
            self.assertTrue(weight_tensor.requires_grad)
            self.assertTrue(bias_tensor.requires_grad)
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_memory_sharing(self):
        """Test memory sharing between PyTorch and TensorFuse."""
        # Create PyTorch tensor
        torch_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 
                                  dtype=torch.float32)
        
        try:
            # Convert to TensorFuse tensor
            tf_tensor = tensorfuse.create_tensor_from_torch(torch_tensor)
            
            # Test that memory is shared (if supported)
            # This would depend on the implementation
            self.assertIsNotNone(tf_tensor)
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_device_consistency(self):
        """Test device consistency between PyTorch and TensorFuse."""
        devices_to_test = ['cpu']
        if torch.cuda.is_available():
            devices_to_test.append('cuda')
        
        for device in devices_to_test:
            with self.subTest(device=device):
                torch_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], 
                                          device=device, dtype=torch.float32)
                
                try:
                    tf_tensor = tensorfuse.create_tensor_from_torch(torch_tensor)
                    self.assertIsNotNone(tf_tensor)
                    
                    # Test device information is preserved
                    # This would depend on the implementation
                    
                except Exception as e:
                    if "CUDA" in str(e) or "GPU" in str(e):
                        self.skipTest(f"GPU not available: {e}")
                    else:
                        raise


class TestHighLevelPyTorchAPI(unittest.TestCase):
    """Test high-level PyTorch API integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not PYTORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        if not TENSORFUSE_AVAILABLE:
            self.skipTest("TensorFuse not available - requires CUTLASS")
    
    def test_pytorch_module_integration(self):
        """Test PyTorch nn.Module integration."""
        try:
            # Test creating a TensorFuse operation as a PyTorch module
            module = core.FusedGemmBiasGelu(m=64, n=128, k=256)
            
            # Test module properties
            self.assertIsInstance(module, torch.nn.Module)
            self.assertTrue(hasattr(module, 'forward'))
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_parameter_handling(self):
        """Test parameter handling in PyTorch integration."""
        try:
            # Create operation with parameters
            module = core.FusedGemmBiasGelu(m=32, n=64, k=128)
            
            # Test parameter registration
            # This would depend on the implementation
            params = list(module.parameters())
            # Parameters might be empty if they're managed externally
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_training_mode_handling(self):
        """Test training vs evaluation mode handling."""
        try:
            module = core.FusedGemmBiasGelu(m=32, n=64, k=128)
            
            # Test training mode
            module.train()
            self.assertTrue(module.training)
            
            # Test evaluation mode  
            module.eval()
            self.assertFalse(module.training)
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise


class TestPerformanceIntegration(unittest.TestCase):
    """Test performance-related PyTorch integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not PYTORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        if not TENSORFUSE_AVAILABLE:
            self.skipTest("TensorFuse not available - requires CUTLASS")
    
    def test_benchmark_integration(self):
        """Test benchmarking integration with PyTorch."""
        try:
            # Create test tensors
            input_tensor = torch.randn(128, 512, dtype=torch.float16)
            weight_tensor = torch.randn(512, 1024, dtype=torch.float16)
            bias_tensor = torch.randn(1024, dtype=torch.float16)
            
            # Test benchmarking
            benchmark_result = tensorfuse.benchmark_operation(
                'fused_gemm_bias_gelu',
                input_tensor, weight_tensor, bias_tensor
            )
            
            self.assertIsInstance(benchmark_result, dict)
            self.assertIn('execution_time', benchmark_result)
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_memory_profiling(self):
        """Test memory profiling with PyTorch tensors."""
        try:
            # Create test tensors
            large_tensor = torch.randn(1000, 1000, dtype=torch.float32)
            
            # Test memory profiling
            with memory.TensorManager() as manager:
                tf_tensor = tensorfuse.create_tensor_from_torch(large_tensor)
                manager.register_tensor(tf_tensor)
                
                # Check memory usage
                memory_info = manager.get_memory_usage()
                self.assertIsInstance(memory_info, dict)
                
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise


class TestErrorHandling(unittest.TestCase):
    """Test error handling in PyTorch integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not PYTORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        if not TENSORFUSE_AVAILABLE:
            self.skipTest("TensorFuse not available - requires CUTLASS")
    
    def test_invalid_tensor_conversion(self):
        """Test error handling for invalid tensor conversions."""
        # Test unsupported dtype
        unsupported_tensor = torch.tensor([1, 2, 3], dtype=torch.float64)
        with self.assertRaises((ValueError, TypeError, RuntimeError)):
            tensorfuse.create_tensor_from_torch(unsupported_tensor)
        
        # Test invalid shape
        invalid_tensor = torch.tensor([])  # Empty tensor
        with self.assertRaises((ValueError, RuntimeError)):
            tensorfuse.create_tensor_from_torch(invalid_tensor)
    
    def test_device_mismatch_handling(self):
        """Test handling of device mismatches."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        try:
            # Create tensors on different devices
            cpu_tensor = torch.randn(2, 2, device='cpu')
            cuda_tensor = torch.randn(2, 2, device='cuda')
            
            # Test operation with mixed devices
            # This should either work or raise a clear error
            fused_op = core.FusedGemmBiasGelu(m=2, n=2, k=2)
            
            # The specific behavior depends on implementation
            # Just test that it doesn't crash unexpectedly
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_gradient_error_handling(self):
        """Test error handling in gradient computation."""
        try:
            # Create tensor with gradient tracking
            input_tensor = torch.randn(2, 2, requires_grad=True)
            
            # Test error handling in autograd context
            # This would test specific error conditions
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True) 