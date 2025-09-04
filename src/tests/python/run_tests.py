#!/usr/bin/env python3
"""
Test runner for TensorFuse Python bindings.

Runs all Python tests and provides a comprehensive summary of results.
"""

import unittest
import sys
import os
import time
from io import StringIO

# Add the Python bindings to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))

# Import test modules
from test_core_bindings import *
from test_numpy_integration import *
from test_pytorch_integration import *
from test_profiling import *


def run_test_suite():
    """Run the complete test suite."""
    
    print("=" * 80)
    print("TensorFuse Python Bindings Test Suite")
    print("=" * 80)
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    
    # Check TensorFuse availability
    try:
        import tensorfuse
        print("✅ TensorFuse available")
        tensorfuse_available = True
    except ImportError as e:
        print(f"❌ TensorFuse not available: {e}")
        print("   This is expected if CUTLASS is not installed")
        tensorfuse_available = False
    
    # Check NumPy availability
    try:
        import numpy as np
        print("✅ NumPy available")
        numpy_available = True
    except ImportError:
        print("❌ NumPy not available")
        numpy_available = False
    
    # Check PyTorch availability
    try:
        import torch
        print("✅ PyTorch available")
        pytorch_available = True
    except ImportError:
        print("❌ PyTorch not available")
        pytorch_available = False
    
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test modules
    test_modules = [
        'test_core_bindings',
        'test_numpy_integration',
        'test_pytorch_integration', 
        'test_profiling'
    ]
    
    for module_name in test_modules:
        try:
            module = sys.modules[module_name]
            suite.addTests(loader.loadTestsFromModule(module))
        except Exception as e:
            print(f"Warning: Could not load tests from {module_name}: {e}")
    
    # Run tests
    print("Running tests...")
    print("-" * 80)
    
    # Custom test runner with detailed output
    stream = StringIO()
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=2,
        buffer=True
    )
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print results
    print(stream.getvalue())
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    passed = total_tests - failures - errors - skipped
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print()
    
    # Detailed failure/error reporting
    if failures:
        print("FAILURES:")
        print("-" * 40)
        for test, traceback in result.failures:
            print(f"❌ {test}")
            print(f"   {traceback.splitlines()[-1]}")
        print()
    
    if errors:
        print("ERRORS:")
        print("-" * 40)
        for test, traceback in result.errors:
            print(f"💥 {test}")
            print(f"   {traceback.splitlines()[-1]}")
        print()
    
    if skipped:
        print("SKIPPED TESTS:")
        print("-" * 40)
        for test, reason in result.skipped:
            print(f"⏭️  {test}")
            print(f"   Reason: {reason}")
        print()
    
    # Status assessment
    if not tensorfuse_available:
        print("📝 NOTE: Most tests were skipped because TensorFuse requires CUTLASS")
        print("   To run full tests, use the Docker environment or install CUTLASS")
        print("   Docker command: docker run --gpus all -it tensorfuse:dev")
    
    # Success/failure status
    if failures == 0 and errors == 0:
        print("✅ ALL TESTS PASSED!")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


def run_specific_test(test_name):
    """Run a specific test module."""
    
    print(f"Running specific test: {test_name}")
    print("-" * 80)
    
    # Create test suite for specific module
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    try:
        module = sys.modules[test_name]
        suite.addTests(loader.loadTestsFromModule(module))
        
        # Run the test
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return 0 if result.wasSuccessful() else 1
        
    except Exception as e:
        print(f"Error running test {test_name}: {e}")
        return 1


def main():
    """Main entry point."""
    
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        return run_specific_test(test_name)
    else:
        # Run all tests
        return run_test_suite()


if __name__ == '__main__':
    sys.exit(main()) 