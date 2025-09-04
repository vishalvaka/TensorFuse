#!/usr/bin/env python3
"""
NumPy integration tests for TensorFuse.

Tests the NumPy array to TensorFuse tensor conversion functionality including:
- Array creation and conversion
- Data type mapping
- Shape validation
- Memory management
- Error handling
"""

import unittest
import sys
import os
import numpy as np

# Add the Python bindings to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))

try:
    import tensorfuse
    from tensorfuse import core, memory, utils
    TENSORFUSE_AVAILABLE = True
except ImportError as e:
    TENSORFUSE_AVAILABLE = False
    print(f"TensorFuse not available: {e}")
    print("This is expected if CUTLASS is not installed")


class TestNumpyIntegration(unittest.TestCase):
    """Test NumPy array integration with TensorFuse."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not TENSORFUSE_AVAILABLE:
            self.skipTest("TensorFuse not available - requires CUTLASS")
    
    def test_numpy_to_tensor_conversion(self):
        """Test NumPy array to TensorFuse tensor conversion."""
        # Test different data types
        test_cases = [
            (np.float32, tensorfuse.TensorFuseDataType.FP32),
            (np.float16, tensorfuse.TensorFuseDataType.FP16),
            (np.int8, tensorfuse.TensorFuseDataType.INT8),
            (np.uint8, tensorfuse.TensorFuseDataType.UINT8),
            (np.int32, tensorfuse.TensorFuseDataType.INT32),
        ]
        
        for np_dtype, tf_dtype in test_cases:
            with self.subTest(np_dtype=np_dtype, tf_dtype=tf_dtype):
                # Create test array
                arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np_dtype)
                
                try:
                    # Convert to TensorFuse tensor
                    tensor = tensorfuse.create_tensor_from_numpy(arr)
                    
                    # Verify conversion succeeded
                    self.assertIsNotNone(tensor)
                    
                except Exception as e:
                    if "CUDA" in str(e) or "GPU" in str(e):
                        self.skipTest(f"GPU not available: {e}")
                    else:
                        raise
    
    def test_bfloat16_conversion(self):
        """Test bfloat16 conversion if available."""
        # bfloat16 might not be available in all NumPy versions
        if not hasattr(np, 'bfloat16'):
            self.skipTest("bfloat16 not available in this NumPy version")
        
        try:
            arr = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.bfloat16)
            tensor = tensorfuse.create_tensor_from_numpy(arr)
            self.assertIsNotNone(tensor)
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            elif "bfloat16" in str(e):
                self.skipTest("bfloat16 not supported in this configuration")
            else:
                raise
    
    def test_shape_validation(self):
        """Test shape validation for NumPy arrays."""
        # Valid shapes
        valid_shapes = [
            (2, 3),
            (4, 4, 4),
            (1, 256, 256),
            (2, 3, 4, 5),
        ]
        
        for shape in valid_shapes:
            with self.subTest(shape=shape):
                arr = np.random.rand(*shape).astype(np.float32)
                try:
                    tensor = tensorfuse.create_tensor_from_numpy(arr)
                    self.assertIsNotNone(tensor)
                except Exception as e:
                    if "CUDA" in str(e) or "GPU" in str(e):
                        self.skipTest(f"GPU not available: {e}")
                    else:
                        raise
    
    def test_invalid_shapes(self):
        """Test error handling for invalid shapes."""
        # Test empty array
        with self.assertRaises((ValueError, RuntimeError)):
            empty_arr = np.array([])
            tensorfuse.create_tensor_from_numpy(empty_arr)
        
        # Test scalar
        with self.assertRaises((ValueError, RuntimeError)):
            scalar = np.array(5.0)
            tensorfuse.create_tensor_from_numpy(scalar)
        
        # Test too many dimensions (if there's a limit)
        try:
            max_dims = tensorfuse.TENSORFUSE_MAX_DIMS
            if max_dims > 0:
                invalid_shape = [2] * (max_dims + 1)
                arr = np.random.rand(*invalid_shape).astype(np.float32)
                with self.assertRaises((ValueError, RuntimeError)):
                    tensorfuse.create_tensor_from_numpy(arr)
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_unsupported_dtypes(self):
        """Test error handling for unsupported data types."""
        unsupported_dtypes = [
            np.float64,
            np.int64,
            np.complex64,
            np.complex128,
            np.bool_,
        ]
        
        for dtype in unsupported_dtypes:
            with self.subTest(dtype=dtype):
                arr = np.array([[1, 2], [3, 4]], dtype=dtype)
                with self.assertRaises((ValueError, TypeError, RuntimeError)):
                    tensorfuse.create_tensor_from_numpy(arr)
    
    def test_memory_layout(self):
        """Test memory layout handling (C-contiguous vs Fortran-contiguous)."""
        # Test C-contiguous array
        c_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32, order='C')
        self.assertTrue(c_array.flags['C_CONTIGUOUS'])
        
        try:
            c_tensor = tensorfuse.create_tensor_from_numpy(c_array)
            self.assertIsNotNone(c_tensor)
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
        
        # Test Fortran-contiguous array
        f_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32, order='F')
        self.assertTrue(f_array.flags['F_CONTIGUOUS'])
        
        try:
            f_tensor = tensorfuse.create_tensor_from_numpy(f_array)
            self.assertIsNotNone(f_tensor)
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_stride_handling(self):
        """Test handling of strided arrays."""
        # Create strided array
        base_array = np.arange(24, dtype=np.float32).reshape(4, 6)
        strided_array = base_array[::2, ::2]  # Every other element
        
        self.assertFalse(strided_array.flags['C_CONTIGUOUS'])
        self.assertFalse(strided_array.flags['F_CONTIGUOUS'])
        
        try:
            tensor = tensorfuse.create_tensor_from_numpy(strided_array)
            self.assertIsNotNone(tensor)
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            elif "contiguous" in str(e).lower():
                # This is expected - strided arrays might not be supported
                pass
            else:
                raise


class TestMemoryManagement(unittest.TestCase):
    """Test memory management with NumPy arrays."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not TENSORFUSE_AVAILABLE:
            self.skipTest("TensorFuse not available - requires CUTLASS")
    
    def test_memory_pool_integration(self):
        """Test MemoryPool integration with NumPy arrays."""
        try:
            pool = memory.MemoryPool(initial_size=1024*1024)  # 1MB
            
            # Test allocation through pool
            arr = np.random.rand(100, 100).astype(np.float32)
            
            with pool:
                tensor = tensorfuse.create_tensor_from_numpy(arr)
                self.assertIsNotNone(tensor)
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_tensor_manager_context(self):
        """Test TensorManager context manager."""
        try:
            manager = memory.TensorManager()
            
            with manager:
                # Create multiple tensors
                arrays = [
                    np.random.rand(50, 50).astype(np.float32),
                    np.random.rand(100, 100).astype(np.float16),
                    np.random.rand(25, 25).astype(np.int32),
                ]
                
                tensors = []
                for arr in arrays:
                    tensor = tensorfuse.create_tensor_from_numpy(arr)
                    tensors.append(tensor)
                    manager.register_tensor(tensor)
                
                # Verify all tensors are registered
                self.assertEqual(len(manager.managed_tensors), len(arrays))
            
            # After context exit, tensors should be cleaned up
            self.assertEqual(len(manager.managed_tensors), 0)
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_large_array_handling(self):
        """Test handling of large NumPy arrays."""
        # Test with a reasonably large array
        large_shape = (1000, 1000)
        large_array = np.random.rand(*large_shape).astype(np.float32)
        
        try:
            tensor = tensorfuse.create_tensor_from_numpy(large_array)
            self.assertIsNotNone(tensor)
            
            # Test memory info
            memory_info = tensorfuse.get_memory_info()
            self.assertIsInstance(memory_info, dict)
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            elif "memory" in str(e).lower() or "allocation" in str(e).lower():
                self.skipTest(f"Insufficient memory: {e}")
            else:
                raise


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for NumPy integration."""
    
    def test_numpy_dtype_conversion(self):
        """Test NumPy dtype to TensorFuse dtype conversion."""
        if not TENSORFUSE_AVAILABLE:
            self.skipTest("TensorFuse not available - requires CUTLASS")
        
        # Test valid conversions
        conversions = [
            (np.float32, tensorfuse.TensorFuseDataType.FP32),
            (np.float16, tensorfuse.TensorFuseDataType.FP16),
            (np.int8, tensorfuse.TensorFuseDataType.INT8),
            (np.uint8, tensorfuse.TensorFuseDataType.UINT8),
            (np.int32, tensorfuse.TensorFuseDataType.INT32),
        ]
        
        for np_dtype, expected_tf_dtype in conversions:
            with self.subTest(np_dtype=np_dtype):
                arr = np.array([1, 2, 3], dtype=np_dtype)
                tf_dtype = utils.numpy_dtype_to_tensorfuse(arr.dtype)
                self.assertEqual(tf_dtype, expected_tf_dtype)
    
    def test_shape_compatibility(self):
        """Test shape compatibility checking."""
        if not TENSORFUSE_AVAILABLE:
            self.skipTest("TensorFuse not available - requires CUTLASS")
        
        # Test compatible shapes
        compatible_shapes = [
            (2, 3),
            (4, 4, 4),
            (1, 256),
            (2, 3, 4, 5),
        ]
        
        for shape in compatible_shapes:
            with self.subTest(shape=shape):
                arr = np.random.rand(*shape).astype(np.float32)
                self.assertTrue(utils.is_shape_compatible(arr.shape))
    
    def test_memory_requirements(self):
        """Test memory requirement calculations."""
        if not TENSORFUSE_AVAILABLE:
            self.skipTest("TensorFuse not available - requires CUTLASS")
        
        # Test memory size calculation
        arr = np.random.rand(100, 100).astype(np.float32)
        expected_size = arr.nbytes
        
        calculated_size = utils.calculate_tensor_size(
            arr.shape, 
            tensorfuse.TensorFuseDataType.FP32
        )
        
        self.assertEqual(calculated_size, expected_size)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True) 