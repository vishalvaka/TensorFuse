#!/bin/bash
set -e

# TensorFuse Test Script
# This script runs all TensorFuse tests with proper error handling

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

# Parse command line arguments
RUN_CPP_TESTS=true
RUN_PYTHON_TESTS=true
VERBOSE=false
STOP_ON_FAIL=false

for arg in "$@"
do
    case $arg in
        --cpp-only)
        RUN_CPP_TESTS=true
        RUN_PYTHON_TESTS=false
        shift
        ;;
        --python-only)
        RUN_CPP_TESTS=false
        RUN_PYTHON_TESTS=true
        shift
        ;;
        --verbose)
        VERBOSE=true
        shift
        ;;
        --stop-on-fail)
        STOP_ON_FAIL=true
        shift
        ;;
        --help)
        echo "Usage: $0 [OPTIONS]"
        echo "Options:"
        echo "  --cpp-only      Run only C++ tests"
        echo "  --python-only   Run only Python tests"
        echo "  --verbose       Verbose output"
        echo "  --stop-on-fail  Stop on first failure"
        echo "  --help          Show this help message"
        exit 0
        ;;
        *)
        print_error "Unknown option: $arg"
        exit 1
        ;;
    esac
done

# Get workspace directory
WORKSPACE_DIR="${WORKSPACE_DIR:-$(pwd)}"
BUILD_DIR="${WORKSPACE_DIR}/build"

print_info "Starting TensorFuse tests..."
print_info "Workspace: ${WORKSPACE_DIR}"
print_info "Build directory: ${BUILD_DIR}"

# Change to workspace directory
cd "${WORKSPACE_DIR}"

# Check if build directory exists
if [ ! -d "${BUILD_DIR}" ]; then
    print_error "Build directory not found. Please run './scripts/build.sh' first."
    exit 1
fi

# Check GPU access
print_info "Checking GPU access..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    print_info "Found ${GPU_COUNT} GPU(s) with ${GPU_MEMORY} memory"
else
    print_warn "nvidia-smi not found, GPU tests may fail"
fi

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test and track results
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    print_section "Running ${test_name}..."
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if [ "$VERBOSE" = true ]; then
        echo "Command: ${test_command}"
    fi
    
    if eval "${test_command}"; then
        print_info "${test_name} PASSED"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        print_error "${test_name} FAILED"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        
        if [ "$STOP_ON_FAIL" = true ]; then
            print_error "Stopping due to test failure"
            exit 1
        fi
        return 1
    fi
}

# C++ Tests
if [ "$RUN_CPP_TESTS" = true ]; then
    print_section "=== C++ Tests ==="
    
    # Check if C++ tests were built
    if [ ! -f "${BUILD_DIR}/src/tests/test_kernels" ]; then
        print_warn "C++ tests not found. Skipping C++ tests."
        print_warn "Run './scripts/build.sh' to build tests."
    else
        # Run CTest
        cd "${BUILD_DIR}"
        
        if [ "$VERBOSE" = true ]; then
            run_test "CTest Suite" "ctest --verbose --output-on-failure"
        else
            run_test "CTest Suite" "ctest --output-on-failure"
        fi
        
        # Run individual tests
        if [ -f "src/tests/test_kernels" ]; then
            run_test "Kernel Tests" "./src/tests/test_kernels"
        fi
        
        if [ -f "src/tests/test_simple_debug" ]; then
            run_test "Simple Debug Test" "./src/tests/test_simple_debug"
        fi
        
        if [ -f "src/tests/test_working_kernel" ]; then
            run_test "Working Kernel Test" "./src/tests/test_working_kernel"
        fi
        
        if [ -f "src/tests/test_kernels_simple" ]; then
            run_test "Simple Kernel Test" "./src/tests/test_kernels_simple"
        fi
        
        cd "${WORKSPACE_DIR}"
    fi
fi

# Python Tests
if [ "$RUN_PYTHON_TESTS" = true ]; then
    print_section "=== Python Tests ==="
    
    # Check if Python tests directory exists
    if [ ! -d "src/tests/python" ]; then
        print_warn "Python tests not found. Skipping Python tests."
    else
        # Check if tensorfuse module is available
        if ! python -c "import tensorfuse" 2>/dev/null; then
            print_warn "TensorFuse Python module not found. Skipping Python tests."
            print_warn "Run './scripts/build.sh' to build Python bindings."
        else
            cd "src/tests/python"
            
            # Run pytest with proper options
            PYTEST_ARGS="-v"
            if [ "$VERBOSE" = true ]; then
                PYTEST_ARGS="${PYTEST_ARGS} -s"
            fi
            
            # Run all Python tests
            if [ "$VERBOSE" = true ]; then
                run_test "Python Test Suite" "python -m pytest ${PYTEST_ARGS} --tb=long"
            else
                run_test "Python Test Suite" "python -m pytest ${PYTEST_ARGS} --tb=short"
            fi
            
            # Run individual test files
            for test_file in test_*.py; do
                if [ -f "$test_file" ]; then
                    test_name=$(basename "$test_file" .py)
                    run_test "Python ${test_name}" "python -m pytest ${test_file} ${PYTEST_ARGS} --tb=short"
                fi
            done
            
            cd "${WORKSPACE_DIR}"
        fi
    fi
fi

# Test Summary
print_section "=== Test Summary ==="
print_info "Total tests: ${TOTAL_TESTS}"
print_info "Passed: ${PASSED_TESTS}"
if [ $FAILED_TESTS -gt 0 ]; then
    print_error "Failed: ${FAILED_TESTS}"
else
    print_info "Failed: ${FAILED_TESTS}"
fi

# Final result
if [ $FAILED_TESTS -eq 0 ]; then
    print_info "All tests passed! ✅"
    exit 0
else
    print_error "Some tests failed! ❌"
    exit 1
fi 