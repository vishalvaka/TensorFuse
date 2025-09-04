#!/usr/bin/env python3
"""
Profiling functionality tests for TensorFuse.

Tests the profiling and benchmarking capabilities including:
- Profile configuration
- Metrics collection
- Benchmarking utilities
- Performance analysis
"""

import unittest
import sys
import os
import time

# Add the Python bindings to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))

try:
    import tensorfuse
    from tensorfuse import profiler, core, memory
    TENSORFUSE_AVAILABLE = True
except ImportError as e:
    TENSORFUSE_AVAILABLE = False
    print(f"TensorFuse not available: {e}")
    print("This is expected if CUTLASS is not installed")


class TestProfilingConfiguration(unittest.TestCase):
    """Test profiling configuration functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not TENSORFUSE_AVAILABLE:
            self.skipTest("TensorFuse not available - requires CUTLASS")
    
    def test_profiling_config(self):
        """Test profiling configuration options."""
        try:
            # Test default configuration
            config = tensorfuse.get_profiling_config()
            self.assertIsInstance(config, dict)
            
            # Test setting configuration
            new_config = {
                'enable_timing': True,
                'enable_memory_tracking': True,
                'enable_kernel_profiling': False
            }
            
            result = tensorfuse.configure_profiling(new_config)
            self.assertEqual(result, tensorfuse.TensorFuseStatus.SUCCESS)
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_profiling_enable_disable(self):
        """Test enabling and disabling profiling."""
        try:
            # Test enabling profiling
            tensorfuse.enable_profiling()
            
            # Test disabling profiling
            tensorfuse.disable_profiling()
            
            # Test querying profiling status
            status = tensorfuse.is_profiling_enabled()
            self.assertIsInstance(status, bool)
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_profiling_modes(self):
        """Test different profiling modes."""
        profiling_modes = [
            'timing_only',
            'memory_only', 
            'full_profiling',
            'kernel_profiling'
        ]
        
        for mode in profiling_modes:
            with self.subTest(mode=mode):
                try:
                    result = tensorfuse.set_profiling_mode(mode)
                    self.assertEqual(result, tensorfuse.TensorFuseStatus.SUCCESS)
                except Exception as e:
                    if "CUDA" in str(e) or "GPU" in str(e):
                        self.skipTest(f"GPU not available: {e}")
                    else:
                        raise


class TestMetricsCollection(unittest.TestCase):
    """Test metrics collection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not TENSORFUSE_AVAILABLE:
            self.skipTest("TensorFuse not available - requires CUTLASS")
    
    def test_basic_metrics(self):
        """Test basic metrics collection."""
        try:
            # Enable profiling
            tensorfuse.enable_profiling()
            
            # Get initial metrics
            initial_metrics = tensorfuse.get_profile_metrics()
            self.assertIsInstance(initial_metrics, dict)
            
            # Test expected metric keys
            expected_keys = ['execution_time', 'memory_usage', 'kernel_count']
            for key in expected_keys:
                if key in initial_metrics:
                    self.assertIsInstance(initial_metrics[key], (int, float))
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        try:
            # Enable profiling
            tensorfuse.enable_profiling()
            
            # Reset metrics
            tensorfuse.reset_profile_metrics()
            
            # Verify metrics are reset
            metrics = tensorfuse.get_profile_metrics()
            self.assertIsInstance(metrics, dict)
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_detailed_metrics(self):
        """Test detailed metrics collection."""
        try:
            # Enable detailed profiling
            tensorfuse.configure_profiling({
                'enable_timing': True,
                'enable_memory_tracking': True,
                'enable_kernel_profiling': True,
                'detailed_metrics': True
            })
            
            # Get detailed metrics
            detailed_metrics = tensorfuse.get_detailed_metrics()
            self.assertIsInstance(detailed_metrics, dict)
            
            # Test detailed metric categories
            expected_categories = ['timing', 'memory', 'kernels', 'operations']
            for category in expected_categories:
                if category in detailed_metrics:
                    self.assertIsInstance(detailed_metrics[category], dict)
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise


class TestBenchmarking(unittest.TestCase):
    """Test benchmarking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not TENSORFUSE_AVAILABLE:
            self.skipTest("TensorFuse not available - requires CUTLASS")
    
    def test_operation_benchmarking(self):
        """Test benchmarking of operations."""
        try:
            # Test benchmarking individual operations
            operations = [
                'fused_gemm_bias_gelu',
                'fused_softmax_dropout',
                'fused_multi_head_attention'
            ]
            
            for op_name in operations:
                with self.subTest(operation=op_name):
                    # Create dummy inputs for benchmarking
                    # This would depend on the specific operation
                    benchmark_result = tensorfuse.benchmark_operation(
                        op_name, 
                        iterations=10,
                        warmup_iterations=5
                    )
                    
                    self.assertIsInstance(benchmark_result, dict)
                    self.assertIn('avg_execution_time', benchmark_result)
                    self.assertIn('min_execution_time', benchmark_result)
                    self.assertIn('max_execution_time', benchmark_result)
                    
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_memory_benchmarking(self):
        """Test memory usage benchmarking."""
        try:
            # Test memory benchmark
            memory_benchmark = tensorfuse.benchmark_memory_usage(
                operation='fused_gemm_bias_gelu',
                input_shapes=[(128, 512), (512, 1024), (1024,)],
                dtype=tensorfuse.TensorFuseDataType.FP16
            )
            
            self.assertIsInstance(memory_benchmark, dict)
            self.assertIn('peak_memory_usage', memory_benchmark)
            self.assertIn('allocated_memory', memory_benchmark)
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_throughput_benchmarking(self):
        """Test throughput benchmarking."""
        try:
            # Test throughput measurement
            throughput_result = tensorfuse.benchmark_throughput(
                operation='fused_gemm_bias_gelu',
                input_shapes=[(128, 512), (512, 1024), (1024,)],
                batch_sizes=[1, 8, 32, 128],
                dtype=tensorfuse.TensorFuseDataType.FP16
            )
            
            self.assertIsInstance(throughput_result, dict)
            self.assertIn('ops_per_second', throughput_result)
            self.assertIn('tokens_per_second', throughput_result)
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise


class TestSimpleProfiler(unittest.TestCase):
    """Test SimpleProfiler class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not TENSORFUSE_AVAILABLE:
            self.skipTest("TensorFuse not available - requires CUTLASS")
    
    def test_simple_profiler_basic(self):
        """Test basic SimpleProfiler functionality."""
        try:
            profiler_instance = profiler.SimpleProfiler()
            
            # Test profiler methods
            self.assertTrue(hasattr(profiler_instance, 'start'))
            self.assertTrue(hasattr(profiler_instance, 'stop'))
            self.assertTrue(hasattr(profiler_instance, 'get_results'))
            
            # Test basic profiling
            profiler_instance.start('test_operation')
            time.sleep(0.01)  # Small delay for testing
            profiler_instance.stop('test_operation')
            
            results = profiler_instance.get_results()
            self.assertIsInstance(results, dict)
            self.assertIn('test_operation', results)
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_profiler_context_manager(self):
        """Test SimpleProfiler as context manager."""
        try:
            profiler_instance = profiler.SimpleProfiler()
            
            # Test context manager usage
            with profiler_instance.profile('context_test'):
                time.sleep(0.01)  # Small delay for testing
            
            results = profiler_instance.get_results()
            self.assertIn('context_test', results)
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_profiler_nested_operations(self):
        """Test profiling nested operations."""
        try:
            profiler_instance = profiler.SimpleProfiler()
            
            # Test nested profiling
            with profiler_instance.profile('outer_operation'):
                time.sleep(0.005)
                with profiler_instance.profile('inner_operation'):
                    time.sleep(0.005)
                time.sleep(0.005)
            
            results = profiler_instance.get_results()
            self.assertIn('outer_operation', results)
            self.assertIn('inner_operation', results)
            
            # Outer operation should take longer than inner
            outer_time = results['outer_operation']['total_time']
            inner_time = results['inner_operation']['total_time']
            self.assertGreater(outer_time, inner_time)
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise


class TestPerformanceAnalysis(unittest.TestCase):
    """Test performance analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not TENSORFUSE_AVAILABLE:
            self.skipTest("TensorFuse not available - requires CUTLASS")
    
    def test_roofline_analysis(self):
        """Test roofline analysis functionality."""
        try:
            # Test roofline analysis
            roofline_data = tensorfuse.analyze_roofline(
                operation='fused_gemm_bias_gelu',
                input_shapes=[(128, 512), (512, 1024), (1024,)],
                dtype=tensorfuse.TensorFuseDataType.FP16
            )
            
            self.assertIsInstance(roofline_data, dict)
            expected_keys = ['arithmetic_intensity', 'peak_performance', 'achieved_performance']
            for key in expected_keys:
                if key in roofline_data:
                    self.assertIsInstance(roofline_data[key], (int, float))
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_performance_comparison(self):
        """Test performance comparison functionality."""
        try:
            # Test performance comparison
            comparison_result = tensorfuse.compare_performance(
                operations=['fused_gemm_bias_gelu', 'cublas_gemm'],
                input_shapes=[(128, 512), (512, 1024), (1024,)],
                dtype=tensorfuse.TensorFuseDataType.FP16
            )
            
            self.assertIsInstance(comparison_result, dict)
            self.assertIn('speedup', comparison_result)
            self.assertIn('efficiency', comparison_result)
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_optimization_suggestions(self):
        """Test optimization suggestions."""
        try:
            # Test optimization analysis
            suggestions = tensorfuse.analyze_optimization_opportunities(
                operation='fused_gemm_bias_gelu',
                input_shapes=[(128, 512), (512, 1024), (1024,)],
                dtype=tensorfuse.TensorFuseDataType.FP16
            )
            
            self.assertIsInstance(suggestions, dict)
            self.assertIn('recommendations', suggestions)
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise


class TestProfilingIntegration(unittest.TestCase):
    """Test profiling integration with other components."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not TENSORFUSE_AVAILABLE:
            self.skipTest("TensorFuse not available - requires CUTLASS")
    
    def test_memory_profiling_integration(self):
        """Test integration with memory management."""
        try:
            # Test profiling with memory pool
            with memory.TensorManager() as manager:
                profiler_instance = profiler.SimpleProfiler()
                
                with profiler_instance.profile('memory_operation'):
                    # Simulate memory operations
                    tensor = tensorfuse.create_tensor(
                        shape=[100, 100],
                        dtype=tensorfuse.TensorFuseDataType.FP32
                    )
                    manager.register_tensor(tensor)
                
                # Check profiling results
                results = profiler_instance.get_results()
                self.assertIn('memory_operation', results)
                
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise
    
    def test_operation_profiling_integration(self):
        """Test integration with core operations."""
        try:
            # Test profiling with core operations
            profiler_instance = profiler.SimpleProfiler()
            
            # Create fused operation
            fused_op = core.FusedGemmBiasGelu(m=64, n=128, k=256)
            
            with profiler_instance.profile('fused_operation'):
                # This would call the actual operation
                pass
            
            results = profiler_instance.get_results()
            self.assertIn('fused_operation', results)
            
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.skipTest(f"GPU not available: {e}")
            else:
                raise


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True) 