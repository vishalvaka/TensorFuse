#!/bin/bash
set -e

# TensorFuse Quick Verification Script
# This script quickly verifies that TensorFuse is working correctly

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
    echo -e "${BLUE}[CHECK]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅${NC} $1"
}

print_fail() {
    echo -e "${RED}❌${NC} $1"
}

# Track verification results
CHECKS_PASSED=0
TOTAL_CHECKS=0

# Function to run a verification check
verify_check() {
    local check_name="$1"
    local check_command="$2"
    
    print_section "Checking: ${check_name}"
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if eval "${check_command}" &>/dev/null; then
        print_success "${check_name}"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
        return 0
    else
        print_fail "${check_name}"
        return 1
    fi
}

echo "🔍 TensorFuse Quick Verification"
echo "================================"
echo

# System checks
print_section "=== System Requirements ==="

verify_check "NVIDIA GPU detected" "nvidia-smi --query-gpu=name --format=csv,noheader"
verify_check "CUDA runtime available" "command -v nvcc"
verify_check "Python 3.8+ available" "python -c 'import sys; assert sys.version_info >= (3, 8)'"
verify_check "Git available" "command -v git"

echo

# Build system checks  
print_section "=== Build System ==="

verify_check "CMake available" "command -v cmake"
verify_check "Make available" "command -v make"
verify_check "GCC/G++ available" "command -v g++"

echo

# TensorFuse installation checks
print_section "=== TensorFuse Installation ==="

if [ -d "build" ]; then
    print_success "Build directory exists"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    print_fail "Build directory missing (run ./scripts/build.sh)"
fi
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

if [ -f "build/src/libtensorfuse.so" ]; then
    print_success "C++ library built"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    print_fail "C++ library missing"
fi
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

# Python module check
print_section "=== Python Integration ==="

if python -c "import tensorfuse" 2>/dev/null; then
    print_success "Python module imports successfully"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
    
    # Test basic functionality
    if python -c "
import tensorfuse
import numpy as np
try:
    tensorfuse.init()
    input_tensor = np.random.randn(2, 4, 8).astype(np.float32)
    weight = np.random.randn(8, 16).astype(np.float32)
    bias = np.random.randn(16).astype(np.float32)
    output = np.zeros((2, 4, 16), dtype=np.float32)
    tensorfuse.fused_gemm_bias_gelu(input_tensor, weight, bias, output)
    tensorfuse.shutdown()
    print('Basic operation successful')
except Exception as e:
    print(f'Basic operation failed: {e}')
    exit(1)
" 2>/dev/null; then
        print_success "Basic operations working"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
        print_fail "Basic operations failing"
    fi
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
else
    print_fail "Python module import failed"
fi
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

echo
print_section "=== Verification Summary ==="

if [ $CHECKS_PASSED -eq $TOTAL_CHECKS ]; then
    echo -e "${GREEN}🎉 All checks passed! TensorFuse is ready to use.${NC}"
    echo
    print_info "Next steps:"
    print_info "  • Run tests: ./scripts/test.sh"
    print_info "  • Run benchmarks: ./scripts/benchmark.sh --quick"
    print_info "  • See examples in: src/tests/python/"
    echo
    exit 0
else
    echo -e "${RED}⚠️  ${CHECKS_PASSED}/${TOTAL_CHECKS} checks passed. Some issues found.${NC}"
    echo
    print_info "Troubleshooting:"
    if [ ! -d "build" ]; then
        print_info "  • Build TensorFuse: ./scripts/build.sh"
    fi
    print_info "  • Check requirements: see README.md prerequisites"
    print_info "  • For help: see TROUBLESHOOTING.md"
    echo
    exit 1
fi 