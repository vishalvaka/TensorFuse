# TensorFuse Python Bindings Test Suite

This directory contains the comprehensive test suite for TensorFuse Python bindings.

## Overview

The test suite validates:
- **Core bindings** - Basic library functionality and C++ integration
- **NumPy integration** - Array conversion and data type handling
- **PyTorch integration** - Tensor compatibility and autograd support
- **Profiling functionality** - Performance measurement and analysis
- **Memory management** - Tensor allocation and cleanup
- **Error handling** - Exception handling and validation

## Test Files

### Core Test Files

| File | Description |
|------|-------------|
| `test_core_bindings.py` | Tests fundamental bindings functionality |
| `test_numpy_integration.py` | Tests NumPy array integration |
| `test_pytorch_integration.py` | Tests PyTorch tensor integration |
| `test_profiling.py` | Tests profiling and benchmarking |
| `run_tests.py` | Main test runner script |

### Test Categories

#### 1. Core Bindings (`test_core_bindings.py`)
- Library import and initialization
- Enum and constant definitions
- Core operation availability
- Memory management functions
- Error handling mechanisms
- High-level API validation

#### 2. NumPy Integration (`test_numpy_integration.py`)
- Array to tensor conversion
- Data type mapping (FP32, FP16, BF16, INT8, etc.)
- Shape validation
- Memory layout handling (C/Fortran contiguous)
- Strided array support
- Large array handling

#### 3. PyTorch Integration (`test_pytorch_integration.py`)
- Tensor conversion between PyTorch and TensorFuse
- Autograd compatibility
- CUDA/CPU device handling
- Memory sharing optimization
- PyTorch nn.Module integration
- Gradient computation support

#### 4. Profiling (`test_profiling.py`)
- Profile configuration
- Metrics collection
- Operation benchmarking
- Memory profiling
- Performance analysis
- Roofline analysis

## Running Tests

### Prerequisites

The test suite requires:
- **Python 3.7+**
- **NumPy** (for array integration tests)
- **PyTorch** (for tensor integration tests, optional)
- **TensorFuse with CUTLASS** (for full functionality)

### Basic Usage

```bash
# Run all tests
python run_tests.py

# Run specific test module
python run_tests.py test_core_bindings
python run_tests.py test_numpy_integration
python run_tests.py test_pytorch_integration
python run_tests.py test_profiling

# Run individual test files
python test_core_bindings.py
python test_numpy_integration.py
```

### Expected Behavior

#### Without CUTLASS (Development Environment)
Most tests will be **skipped** with messages like:
```
TensorFuse not available: No module named '_tensorfuse'
This is expected if CUTLASS is not installed
```

This is **normal behavior** - the tests are designed to detect missing dependencies gracefully.

#### With CUTLASS (Docker Environment)
```bash
# Use Docker environment for full testing
docker run --gpus all -it tensorfuse:dev
cd /workspace
python src/tests/python/run_tests.py
```

In this environment, tests will run fully and validate:
- Library loading and initialization
- Core operation functionality
- Memory management
- Array/tensor conversions
- Performance profiling

### Test Output

#### Successful Run (with CUTLASS)
```
================================================================================
TensorFuse Python Bindings Test Suite
================================================================================

Checking dependencies...
✅ TensorFuse available
✅ NumPy available
✅ PyTorch available

Running tests...
--------------------------------------------------------------------------------
test_library_import (test_core_bindings.TestCoreBindings) ... ok
test_numpy_to_tensor_conversion (test_numpy_integration.TestNumpyIntegration) ... ok
test_pytorch_to_tensorfuse_conversion (test_pytorch_integration.TestPyTorchIntegration) ... ok
...

================================================================================
TEST SUMMARY
================================================================================
Total tests: 45
Passed: 45
Failed: 0
Errors: 0
Skipped: 0
Execution time: 2.34 seconds

✅ ALL TESTS PASSED!
```

#### Run without CUTLASS (Development Environment)
```
================================================================================
TensorFuse Python Bindings Test Suite
================================================================================

Checking dependencies...
❌ TensorFuse not available: No module named '_tensorfuse'
   This is expected if CUTLASS is not installed
✅ NumPy available
✅ PyTorch available

Running tests...
--------------------------------------------------------------------------------
test_library_import (test_core_bindings.TestCoreBindings) ... SKIP (TensorFuse not available)
...

================================================================================
TEST SUMMARY
================================================================================
Total tests: 45
Passed: 0
Failed: 0
Errors: 0
Skipped: 45
Execution time: 0.12 seconds

📝 NOTE: Most tests were skipped because TensorFuse requires CUTLASS
   To run full tests, use the Docker environment or install CUTLASS
   Docker command: docker run --gpus all -it tensorfuse:dev

✅ ALL TESTS PASSED!
```

## Test Design Principles

### 1. Graceful Dependency Handling
- Tests check for TensorFuse availability before running
- Missing dependencies result in skipped tests, not failures
- Clear messages explain why tests are skipped

### 2. GPU Environment Awareness
- Tests detect CUDA availability
- GPU-specific tests are skipped in CPU-only environments
- Error messages distinguish between dependency and GPU issues

### 3. Comprehensive Coverage
- Tests cover both success and failure scenarios
- Edge cases and error conditions are validated
- Performance characteristics are verified

### 4. Isolated Test Cases
- Each test is independent and self-contained
- Tests clean up resources after execution
- No shared state between test cases

## Integration with CI/CD

The test suite is designed for integration with continuous integration:

```yaml
# Example GitHub Actions workflow
- name: Run Python Tests
  run: |
    cd src/tests/python
    python run_tests.py
```

### Exit Codes
- **0**: All tests passed (or skipped due to missing dependencies)
- **1**: One or more tests failed or had errors

## Development Workflow

### Adding New Tests

1. **Create test file** following the naming convention `test_*.py`
2. **Import required modules** and handle ImportError gracefully
3. **Use unittest.TestCase** as the base class
4. **Add setUp()** method to check dependencies
5. **Use skipTest()** for missing dependencies
6. **Add to run_tests.py** import list

### Example Test Structure

```python
import unittest
import sys
import os

# Add bindings to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))

try:
    import tensorfuse
    TENSORFUSE_AVAILABLE = True
except ImportError:
    TENSORFUSE_AVAILABLE = False

class TestNewFeature(unittest.TestCase):
    def setUp(self):
        if not TENSORFUSE_AVAILABLE:
            self.skipTest("TensorFuse not available - requires CUTLASS")
    
    def test_feature(self):
        # Test implementation
        pass
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure Python path includes `src/python`
   - Check that CUTLASS is installed (for full functionality)

2. **GPU Tests Failing**
   - Verify CUDA installation
   - Check GPU availability with `nvidia-smi`
   - Ensure proper Docker GPU passthrough

3. **Memory Issues**
   - Tests may be skipped due to insufficient GPU memory
   - This is normal behavior in resource-constrained environments

### Debug Mode

Run tests with verbose output:
```bash
python -m unittest discover -v
```

## Documentation

- **API Documentation**: [TensorFuse Python API](../../python/README.md)
- **C++ Backend**: [Core Library Tests](../README.md)
- **Docker Environment**: [Development Setup](../../../docker/README.md)

## Contributing

When adding new Python bindings:
1. Add corresponding tests to appropriate test file
2. Update test runner if needed
3. Ensure tests handle missing dependencies gracefully
4. Test both with and without CUTLASS installation 