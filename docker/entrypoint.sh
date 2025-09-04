#!/bin/bash

# ==============================================================================
# TensorFuse Docker Entrypoint Script
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check CUDA availability
check_cuda() {
    print_info "Checking CUDA availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA driver detected"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
    else
        print_error "NVIDIA driver not found"
        return 1
    fi
    
    if command -v nvcc &> /dev/null; then
        print_success "CUDA compiler detected"
        nvcc --version | grep "release"
    else
        print_error "CUDA compiler not found"
        return 1
    fi
    
    return 0
}

# Check GPU compute capability
check_gpu_capability() {
    print_info "Checking GPU compute capability..."
    
    python3 -c "
import subprocess
import sys

try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        capabilities = result.stdout.strip().split('\n')
        for cap in capabilities:
            major, minor = map(int, cap.split('.'))
            if major >= 8:  # Ampere or newer
                print(f'GPU compute capability: {cap} (supported)')
            else:
                print(f'GPU compute capability: {cap} (not supported, requires 8.0+)')
                sys.exit(1)
    else:
        print('Could not query GPU compute capability')
        sys.exit(1)
except Exception as e:
    print(f'Error checking GPU capability: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_success "GPU compute capability check passed"
    else
        print_error "GPU compute capability check failed"
        return 1
    fi
}

# Initialize TensorFuse environment
init_tensorfuse() {
    print_info "Initializing TensorFuse environment..."
    
    # Set up environment variables
    export TENSORFUSE_HOME="/workspace"
    export TENSORFUSE_CACHE_DIR="/workspace/cache"
    export TENSORFUSE_BUILD_DIR="/workspace/build"
    
    # Create necessary directories
    mkdir -p ${TENSORFUSE_CACHE_DIR}/autotuner
    mkdir -p ${TENSORFUSE_CACHE_DIR}/kernels
    mkdir -p ${TENSORFUSE_BUILD_DIR}
    mkdir -p /workspace/benchmarks/results
    
    # Check if CMakeLists.txt exists
    if [ -f "/workspace/CMakeLists.txt" ]; then
        print_success "TensorFuse project detected"
    else
        print_warning "TensorFuse project not found - development mode"
    fi
    
    # Configure CMake build if not already configured
    if [ ! -f "${TENSORFUSE_BUILD_DIR}/CMakeCache.txt" ]; then
        print_info "Configuring CMake build..."
        cd /workspace
        cmake -B build -G Ninja \
            -DCMAKE_BUILD_TYPE=${TENSORFUSE_BUILD_TYPE:-Release} \
            -DCMAKE_CUDA_ARCHITECTURES=${TENSORFUSE_CUDA_ARCHITECTURES:-"89;90"} \
            -DTENSORFUSE_ENABLE_PROFILING=${TENSORFUSE_ENABLE_PROFILING:-ON} \
            -DTENSORFUSE_ENABLE_FP8=${TENSORFUSE_ENABLE_FP8:-ON} \
            -DTENSORFUSE_BUILD_TESTS=ON \
            -DTENSORFUSE_BUILD_PYTHON=ON \
            -DTENSORFUSE_BUILD_BENCHMARKS=ON
        
        if [ $? -eq 0 ]; then
            print_success "CMake configuration completed"
        else
            print_error "CMake configuration failed"
            return 1
        fi
    else
        print_info "CMake already configured"
    fi
}

# Start Jupyter Lab if requested
start_jupyter() {
    if [ "$TENSORFUSE_START_JUPYTER" = "true" ]; then
        print_info "Starting Jupyter Lab..."
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
            --NotebookApp.token='' --NotebookApp.password='' \
            --notebook-dir=/workspace/notebooks &
        print_success "Jupyter Lab started on port 8888"
    fi
}

# Print welcome message
print_welcome() {
    echo
    echo "==============================================="
    echo "       TensorFuse Development Environment      "
    echo "==============================================="
    echo
    echo "CUDA Version: $(nvcc --version | grep release | awk '{print $6}' | cut -c2-)"
    echo "GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ' | sed 's/,$//')"
    echo "Workspace: /workspace"
    echo "Build Directory: ${TENSORFUSE_BUILD_DIR}"
    echo "Cache Directory: ${TENSORFUSE_CACHE_DIR}"
    echo
    echo "Quick Commands:"
    echo "  build    - Build TensorFuse library"
    echo "  test     - Run tests"
    echo "  clean    - Clean build directory"
    echo
    echo "Documentation: https://tensorfuse.readthedocs.io"
    echo "==============================================="
    echo
}

# Main execution
main() {
    print_info "Starting TensorFuse container..."
    
    # Check CUDA availability
    if ! check_cuda; then
        print_error "CUDA check failed - GPU operations may not work"
        if [ "$TENSORFUSE_REQUIRE_CUDA" = "true" ]; then
            exit 1
        fi
    fi
    
    # Check GPU compute capability
    if ! check_gpu_capability; then
        print_error "GPU capability check failed"
        if [ "$TENSORFUSE_REQUIRE_CUDA" = "true" ]; then
            exit 1
        fi
    fi
    
    # Initialize TensorFuse environment
    if ! init_tensorfuse; then
        print_error "TensorFuse initialization failed"
        exit 1
    fi
    
    # Start Jupyter Lab if requested
    start_jupyter
    
    # Print welcome message
    print_welcome
    
    # Execute the command passed to the container
    if [ "$#" -eq 0 ]; then
        # No command provided, start interactive bash
        exec /bin/bash
    else
        # Execute the provided command
        exec "$@"
    fi
}

# Run main function
main "$@" 