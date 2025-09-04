#!/usr/bin/env python3
"""
Master test runner for all TensorFuse mathematical verification tests.

This script builds and runs all C++ and Python tests, providing a comprehensive
summary of mathematical correctness verification results.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

def run_command(cmd, cwd=None, timeout=300):
    """Run a command and return (success, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=cwd,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n{'-'*40}")
    print(f"{title}")
    print(f"{'-'*40}")

def build_project():
    """Build the TensorFuse project."""
    print_section("BUILDING TENSORFUSE PROJECT")
    
    # Change to project root
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    # Create build directory
    build_dir = project_root / "build"
    build_dir.mkdir(exist_ok=True)
    
    # Run CMake
    print("Running CMake configuration...")
    success, stdout, stderr = run_command("cmake ..", cwd=build_dir)
    if not success:
        print(f"❌ CMake failed: {stderr}")
        return False
    print("✅ CMake configuration successful")
    
    # Build project
    print("Building project...")
    success, stdout, stderr = run_command("make -j8", cwd=build_dir)
    if not success:
        print(f"❌ Build failed: {stderr}")
        return False
    print("✅ Build successful")
    
    return True

def run_cpp_tests():
    """Run all C++ mathematical verification tests."""
    print_section("C++ MATHEMATICAL VERIFICATION TESTS")
    
    project_root = Path(__file__).parent.parent.parent
    tests_dir = project_root / "build" / "src" / "tests"
    
    # List of C++ test executables
    cpp_tests = [
        ("test_kernels", "Core kernel mathematical verification"),
        ("test_softmax_dropout", "Softmax + Dropout mathematical verification"),
        ("test_gemm_edge_cases", "GEMM edge cases and stress tests"),
        ("test_working_kernel", "Working kernel validation"),
        ("test_kernels_simple", "Simple kernel library test")
    ]
    
    results = {}
    
    for test_name, description in cpp_tests:
        print_subsection(f"{test_name}: {description}")
        
        test_path = tests_dir / test_name
        if not test_path.exists():
            print(f"❌ Test executable not found: {test_path}")
            results[test_name] = {"success": False, "error": "Executable not found"}
            continue
        
        # Run test
        success, stdout, stderr = run_command(str(test_path))
        
        if success:
            print(f"✅ {test_name} PASSED")
            print(f"Output: {stdout}")
            results[test_name] = {"success": True, "output": stdout}
        else:
            print(f"❌ {test_name} FAILED")
            print(f"Error: {stderr}")
            results[test_name] = {"success": False, "error": stderr}
    
    return results

def run_python_tests():
    """Run all Python mathematical verification tests."""
    print_section("PYTHON MATHEMATICAL VERIFICATION TESTS")
    
    project_root = Path(__file__).parent.parent.parent
    python_tests_dir = project_root / "src" / "tests" / "python"
    
    # Set up Python path
    python_path = str(project_root / "src" / "python")
    build_lib_path = str(project_root / "build" / "src")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{python_path}:{env.get('PYTHONPATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{build_lib_path}:{env.get('LD_LIBRARY_PATH', '')}"
    
    # List of Python test files
    python_tests = [
        ("test_mathematical_verification.py", "End-to-end mathematical verification"),
        ("test_core_bindings.py", "Core bindings functionality"),
        ("test_numpy_integration.py", "NumPy integration tests"),
        ("run_tests.py", "Complete Python test suite")
    ]
    
    results = {}
    
    for test_file, description in python_tests:
        print_subsection(f"{test_file}: {description}")
        
        test_path = python_tests_dir / test_file
        if not test_path.exists():
            print(f"❌ Test file not found: {test_path}")
            results[test_file] = {"success": False, "error": "Test file not found"}
            continue
        
        # Run test
        cmd = f"python3 {test_path}"
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                cwd=python_tests_dir,
                env=env,
                timeout=300
            )
            success = result.returncode == 0
            stdout = result.stdout
            stderr = result.stderr
        except Exception as e:
            success = False
            stdout = ""
            stderr = str(e)
        
        if success:
            print(f"✅ {test_file} PASSED")
            print(f"Output: {stdout[-500:]}")  # Show last 500 chars
            results[test_file] = {"success": True, "output": stdout}
        else:
            print(f"❌ {test_file} FAILED")
            print(f"Error: {stderr}")
            results[test_file] = {"success": False, "error": stderr}
    
    return results

def run_performance_tests():
    """Run performance and stress tests."""
    print_section("PERFORMANCE AND STRESS TESTS")
    
    project_root = Path(__file__).parent.parent.parent
    build_dir = project_root / "build"
    
    # Test large matrices
    print_subsection("Large Matrix Test")
    
    # Create a simple performance test
    perf_test_code = """
#include <iostream>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

// Forward declarations
typedef enum {
    TENSORFUSE_SUCCESS = 0,
    TENSORFUSE_ERROR_CUDA_ERROR,
    TENSORFUSE_ERROR_KERNEL_LAUNCH_FAILED,
    TENSORFUSE_ERROR_UNSUPPORTED_OPERATION
} TensorFuseStatus;

namespace tensorfuse {
namespace kernels {
    TensorFuseStatus fused_gemm_bias_gelu_fp32(
        const void* A, const void* B, const void* bias, void* C,
        int M, int N, int K, float alpha, float beta, void* stream);
}
}

int main() {
    // Large matrix test
    const int M = 512, N = 1024, K = 2048;
    
    // Allocate memory
    float *d_A, *d_B, *d_bias, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_bias, N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // Initialize with dummy data
    cudaMemset(d_A, 0, M * K * sizeof(float));
    cudaMemset(d_B, 0, K * N * sizeof(float));
    cudaMemset(d_bias, 0, N * sizeof(float));
    
    // Time the kernel
    auto start = std::chrono::high_resolution_clock::now();
    
    TensorFuseStatus status = tensorfuse::kernels::fused_gemm_bias_gelu_fp32(
        d_A, d_B, d_bias, d_C, M, N, K, 1.0f, 1.0f, nullptr);
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    if (status == TENSORFUSE_SUCCESS) {
        std::cout << "Large matrix test (512x1024x2048) PASSED" << std::endl;
        std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;
    } else {
        std::cout << "Large matrix test FAILED with status: " << status << std::endl;
    }
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_bias);
    cudaFree(d_C);
    
    return status == TENSORFUSE_SUCCESS ? 0 : 1;
}
"""
    
    # Write and compile performance test
    perf_test_file = build_dir / "perf_test.cu"
    with open(perf_test_file, "w") as f:
        f.write(perf_test_code)
    
    # Compile performance test
    cmd = f"nvcc -o {build_dir}/perf_test {perf_test_file} -L{build_dir}/src -ltensorfuse -lcudart"
    success, stdout, stderr = run_command(cmd, cwd=build_dir)
    
    if success:
        # Run performance test
        success, stdout, stderr = run_command(f"{build_dir}/perf_test")
        if success:
            print(f"✅ Performance test PASSED")
            print(f"Output: {stdout}")
        else:
            print(f"❌ Performance test FAILED")
            print(f"Error: {stderr}")
    else:
        print(f"❌ Performance test compilation failed: {stderr}")
    
    return {"performance_test": {"success": success, "output": stdout if success else stderr}}

def generate_summary(cpp_results, python_results, perf_results):
    """Generate a comprehensive test summary."""
    print_section("COMPREHENSIVE TEST SUMMARY")
    
    total_tests = len(cpp_results) + len(python_results) + len(perf_results)
    passed_tests = sum(1 for r in cpp_results.values() if r["success"]) + \
                   sum(1 for r in python_results.values() if r["success"]) + \
                   sum(1 for r in perf_results.values() if r["success"])
    
    print(f"Total tests run: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    print_subsection("C++ Test Results")
    for test_name, result in cpp_results.items():
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print_subsection("Python Test Results")
    for test_name, result in python_results.items():
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print_subsection("Performance Test Results")
    for test_name, result in perf_results.items():
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    # Critical test status
    print_subsection("Critical Mathematical Verification Status")
    
    critical_tests = [
        ("test_kernels", "Core GEMM+Bias+GELU mathematical correctness"),
        ("test_softmax_dropout", "Softmax+Dropout mathematical correctness"),
        ("test_mathematical_verification.py", "End-to-end Python API verification")
    ]
    
    for test_name, description in critical_tests:
        result = cpp_results.get(test_name) or python_results.get(test_name)
        if result:
            status = "✅ VERIFIED" if result["success"] else "❌ FAILED"
            print(f"  {description}: {status}")
    
    # Next steps recommendation
    print_subsection("Next Steps Recommendation")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Mathematical verification is complete.")
        print("✅ Ready to proceed with:")
        print("   - FP16 kernel optimization")
        print("   - Multi-head attention implementation")
        print("   - Performance optimization")
        print("   - Production deployment preparation")
    else:
        print("⚠️  Some tests failed. Recommended actions:")
        print("   1. Fix failing mathematical verification tests")
        print("   2. Investigate kernel implementation issues")
        print("   3. Verify CUDA/hardware compatibility")
        print("   4. Re-run tests after fixes")
    
    return passed_tests == total_tests

def main():
    """Main test runner function."""
    print("TensorFuse Mathematical Verification Test Suite")
    print("=" * 60)
    
    start_time = time.time()
    
    # Build project
    if not build_project():
        print("❌ Build failed, cannot run tests")
        return 1
    
    # Run all test categories
    cpp_results = run_cpp_tests()
    python_results = run_python_tests()
    perf_results = run_performance_tests()
    
    # Generate summary
    all_passed = generate_summary(cpp_results, python_results, perf_results)
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 