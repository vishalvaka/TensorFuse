#!/usr/bin/env python3
"""
Core Python bindings tests for TensorFuse.

Tests the fundamental bindings functionality including:
- Basic library loading
- Enum and constant definitions
- Core operation bindings
- Error handling
"""

import unittest
import sys
import os

# Add the Python bindings to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))

try:
    import tensorfuse
    from tensorfuse import core, utils
    TENSORFUSE_AVAILABLE = True
except ImportError as e:
    TENSORFUSE_AVAILABLE = False
    print(f"TensorFuse not available: {e}")
    print("This is expected if CUTLASS is not installed")


class TestCoreBindings(unittest.TestCase):
    """Test core Python bindings functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not TENSORFUSE_AVAILABLE:
            self.skipTest("TensorFuse not available - requires CUTLASS")
    
    def test_library_import(self):
        """Test that the library can be imported successfully."""
        self.assertTrue(hasattr(tensorfuse, '__version__'))
        self.assertIsInstance(tensorfuse.__version__, str)
    
    def test_status_enum(self):
        """Test TensorFuseStatus enum availability."""
        self.assertTrue(hasattr(tensorfuse, 'TensorFuseStatus'))
        
        # Test enum values
        status = tensorfuse.TensorFuseStatus
        self.assertTrue(hasattr(status, 'SUCCESS'))
        self.assertTrue(hasattr(status, 'ERROR_INVALID_ARGUMENT'))
        self.assertTrue(hasattr(status, 'ERROR_CUDA_ERROR'))
        self.assertTrue(hasattr(status, 'ERROR_OUT_OF_MEMORY'))
    
    def test_data_type_enum(self):
        """Test TensorFuseDataType enum availability."""
        self.assertTrue(hasattr(tensorfuse, 'TensorFuseDataType'))
        
        # Test data type values
        dtype = tensorfuse.TensorFuseDataType
        self.assertTrue(hasattr(dtype, 'FP32'))
        self.assertTrue(hasattr(dtype, 'FP16'))
        self.assertTrue(hasattr(dtype, 'BF16'))
        self.assertTrue(hasattr(dtype, 'INT8'))
        self.assertTrue(hasattr(dtype, 'UINT8'))
        self.assertTrue(hasattr(dtype, 'INT32'))
    
    def test_constants(self):
        """Test that important constants are defined."""
        self.assertTrue(hasattr(tensorfuse, 'TENSORFUSE_MAX_DIMS'))
        self.assertIsInstance(tensorfuse.TENSORFUSE_MAX_DIMS, int)
        self.assertGreaterEqual(tensorfuse.TENSORFUSE_MAX_DIMS, 4)
    
    def test_initialization(self):
        """Test library initialization."""
        # Test that initialization function exists
        self.assertTrue(hasattr(tensorfuse, 'initialize'))
        
        # Test initialization (should not throw)
        try:
            result = tensorfuse.initialize()
            self.assertIsInstance(result, tensorfuse.TensorFuseStatus)
        except Exception as e:
            # GPU not available in test environment is acceptable
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_core_operations_exist(self):
        """Test that core operations are available."""
        # Test fused operations
        self.assertTrue(hasattr(tensorfuse, 'fused_gemm_bias_gelu'))
        self.assertTrue(hasattr(tensorfuse, 'fused_softmax_dropout'))
        self.assertTrue(hasattr(tensorfuse, 'fused_multi_head_attention'))
        
        # Test functions are callable
        self.assertTrue(callable(tensorfuse.fused_gemm_bias_gelu))
        self.assertTrue(callable(tensorfuse.fused_softmax_dropout))
        self.assertTrue(callable(tensorfuse.fused_multi_head_attention))
    
    def test_tensor_creation(self):
        """Test tensor creation functions."""
        self.assertTrue(hasattr(tensorfuse, 'create_tensor'))
        self.assertTrue(hasattr(tensorfuse, 'create_tensor_from_numpy'))
        self.assertTrue(callable(tensorfuse.create_tensor))
        self.assertTrue(callable(tensorfuse.create_tensor_from_numpy))
    
    def test_memory_management(self):
        """Test memory management functions."""
        self.assertTrue(hasattr(tensorfuse, 'allocate_tensor'))
        self.assertTrue(hasattr(tensorfuse, 'deallocate_tensor'))
        self.assertTrue(hasattr(tensorfuse, 'get_memory_info'))
        
        # Test functions are callable
        self.assertTrue(callable(tensorfuse.allocate_tensor))
        self.assertTrue(callable(tensorfuse.deallocate_tensor))
        self.assertTrue(callable(tensorfuse.get_memory_info))
    
    def test_profiling_bindings(self):
        """Test profiling functionality bindings."""
        self.assertTrue(hasattr(tensorfuse, 'configure_profiling'))
        self.assertTrue(hasattr(tensorfuse, 'get_profile_metrics'))
        self.assertTrue(hasattr(tensorfuse, 'benchmark_operation'))
        
        # Test functions are callable
        self.assertTrue(callable(tensorfuse.configure_profiling))
        self.assertTrue(callable(tensorfuse.get_profile_metrics))
        self.assertTrue(callable(tensorfuse.benchmark_operation))
    
    def test_error_handling(self):
        """Test error handling mechanisms."""
        # Test invalid arguments trigger proper errors
        with self.assertRaises((ValueError, TypeError, RuntimeError)):
            tensorfuse.create_tensor(shape=[-1, 2, 3])  # Invalid shape
        
        with self.assertRaises((ValueError, TypeError)):
            tensorfuse.create_tensor(shape=[2, 3], dtype="invalid_type")  # Invalid dtype


class TestHighLevelAPI(unittest.TestCase):
    """Test high-level Python API functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not TENSORFUSE_AVAILABLE:
            self.skipTest("TensorFuse not available - requires CUTLASS")
    
    def test_fused_gemm_bias_gelu_class(self):
        """Test FusedGemmBiasGelu high-level class."""
        try:
            op = core.FusedGemmBiasGelu(
                m=128, n=256, k=512,
                dtype=tensorfuse.TensorFuseDataType.FP16
            )
            self.assertIsInstance(op, core.FusedGemmBiasGelu)
            self.assertEqual(op.m, 128)
            self.assertEqual(op.n, 256)
            self.assertEqual(op.k, 512)
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_validation_utilities(self):
        """Test validation utility functions."""
        # Test shape validation
        self.assertTrue(utils.validate_shape([1, 2, 3, 4]))
        self.assertFalse(utils.validate_shape([1, -2, 3]))
        self.assertFalse(utils.validate_shape([]))
        
        # Test dtype validation
        self.assertTrue(utils.validate_dtype(tensorfuse.TensorFuseDataType.FP32))
        self.assertTrue(utils.validate_dtype('FP32'))
        self.assertFalse(utils.validate_dtype('INVALID'))
        
        # Test dimension validation
        self.assertTrue(utils.validate_dimensions([1, 2, 3]))
        self.assertFalse(utils.validate_dimensions([1, 2, 3, 4, 5, 6]))  # Too many dims
    
    def test_dtype_conversion(self):
        """Test data type conversion utilities."""
        # Test string to enum conversion
        self.assertEqual(
            utils.dtype_from_string('FP32'),
            tensorfuse.TensorFuseDataType.FP32
        )
        
        # Test enum to string conversion
        self.assertEqual(
            utils.dtype_to_string(tensorfuse.TensorFuseDataType.FP16),
            'FP16'
        )
        
        # Test invalid conversions
        with self.assertRaises(ValueError):
            utils.dtype_from_string('INVALID')


class TestModuleStructure(unittest.TestCase):
    """Test module structure and organization."""
    
    def test_module_imports(self):
        """Test that all expected modules can be imported."""
        if not TENSORFUSE_AVAILABLE:
            self.skipTest("TensorFuse not available - requires CUTLASS")
        
        # Test submodule imports
        from tensorfuse import core, memory, profiler, utils
        
        # Test each module has expected attributes
        self.assertTrue(hasattr(core, 'FusedGemmBiasGelu'))
        self.assertTrue(hasattr(memory, 'MemoryPool'))
        self.assertTrue(hasattr(profiler, 'SimpleProfiler'))
        self.assertTrue(hasattr(utils, 'validate_shape'))
    
    def test_version_info(self):
        """Test version information."""
        if not TENSORFUSE_AVAILABLE:
            self.skipTest("TensorFuse not available - requires CUTLASS")
        
        self.assertTrue(hasattr(tensorfuse, '__version__'))
        self.assertIsInstance(tensorfuse.__version__, str)
        
        # Version should be in semantic versioning format
        version_parts = tensorfuse.__version__.split('.')
        self.assertGreaterEqual(len(version_parts), 2)
        self.assertTrue(all(part.isdigit() for part in version_parts[:2]))


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True) 