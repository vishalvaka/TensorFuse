#!/bin/bash
set -e

# TensorFuse Build Script
# This script builds the TensorFuse library with proper error handling

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Parse command line arguments
CLEAN_BUILD=false
BUILD_TYPE="Release"
ENABLE_PROFILING="OFF"
ENABLE_TESTING="ON"
ENABLE_PYTHON="ON"
PARALLEL_JOBS=$(nproc)

for arg in "$@"
do
    case $arg in
        --clean)
        CLEAN_BUILD=true
        shift
        ;;
        --debug)
        BUILD_TYPE="Debug"
        shift
        ;;
        --profile)
        ENABLE_PROFILING="ON"
        shift
        ;;
        --no-tests)
        ENABLE_TESTING="OFF"
        shift
        ;;
        --no-python)
        ENABLE_PYTHON="OFF"
        shift
        ;;
        -j*)
        PARALLEL_JOBS="${arg#-j}"
        shift
        ;;
        --help)
        echo "Usage: $0 [OPTIONS]"
        echo "Options:"
        echo "  --clean      Clean build (remove build directory)"
        echo "  --debug      Build in debug mode"
        echo "  --profile    Enable profiling support"
        echo "  --no-tests   Disable building tests"
        echo "  --no-python  Disable building Python bindings"
        echo "  -j<N>        Use N parallel jobs (default: $(nproc))"
        echo "  --help       Show this help message"
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

print_info "Starting TensorFuse build..."
print_info "Workspace: ${WORKSPACE_DIR}"
print_info "Build directory: ${BUILD_DIR}"
print_info "Build type: ${BUILD_TYPE}"
print_info "Profiling: ${ENABLE_PROFILING}"
print_info "Testing: ${ENABLE_TESTING}"
print_info "Python bindings: ${ENABLE_PYTHON}"
print_info "Parallel jobs: ${PARALLEL_JOBS}"

# Change to workspace directory
cd "${WORKSPACE_DIR}"

# Clean build if requested
if [ "$CLEAN_BUILD" = true ]; then
    print_info "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Check for CUDA
print_info "Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
    print_info "Found CUDA version: ${CUDA_VERSION}"
else
    print_error "CUDA not found. Please install CUDA 12.6 or later."
    exit 1
fi

# Check for CUTLASS
print_info "Checking CUTLASS installation..."
if [ -d "/usr/local/include/cutlass" ]; then
    print_info "Found CUTLASS installation"
else
    print_error "CUTLASS not found. Please install CUTLASS 3.5 or later."
    exit 1
fi

# Detect GPU architecture
print_info "Detecting GPU architecture..."
if command -v nvidia-smi &> /dev/null; then
    GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
    print_info "Detected GPU architecture: ${GPU_ARCH}"
    CMAKE_CUDA_ARCHITECTURES="${GPU_ARCH}"
else
    print_warn "nvidia-smi not found, using default architecture 89"
    CMAKE_CUDA_ARCHITECTURES="89"
fi

# Configure CMake
print_info "Configuring CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} \
    -DTENSORFUSE_ENABLE_PROFILING=${ENABLE_PROFILING} \
    -DTENSORFUSE_BUILD_TESTS=${ENABLE_TESTING} \
    -DTENSORFUSE_BUILD_PYTHON=${ENABLE_PYTHON} \
    -DTENSORFUSE_BUILD_BENCHMARKS=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

if [ $? -ne 0 ]; then
    print_error "CMake configuration failed!"
    exit 1
fi

# Build the project
print_info "Building TensorFuse (using ${PARALLEL_JOBS} parallel jobs)..."
make -j${PARALLEL_JOBS}

if [ $? -ne 0 ]; then
    print_error "Build failed!"
    exit 1
fi

# Deploy Python module if Python bindings are enabled
if [ "$ENABLE_PYTHON" = "ON" ]; then
    print_info "Deploying Python module..."
    cd "${WORKSPACE_DIR}"
    
    # Run the deployment script
    if [ -f "scripts/deploy_python_module.sh" ]; then
        ./scripts/deploy_python_module.sh
        if [ $? -ne 0 ]; then
            print_warn "Python module deployment failed, but library build was successful"
            print_warn "You can manually deploy with: ./scripts/deploy_python_module.sh"
        fi
    else
        print_warn "Python deployment script not found"
    fi
fi

print_info "Build completed successfully!"

# Show build summary
print_info "Build Summary:"
print_info "  Build Type: ${BUILD_TYPE}"
print_info "  CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}"
print_info "  Profiling: ${ENABLE_PROFILING}"
print_info "  Tests: ${ENABLE_TESTING}"
print_info "  Python: ${ENABLE_PYTHON}"

# Show next steps
print_info "Next steps:"
if [ "$ENABLE_TESTING" = "ON" ]; then
    print_info "  Run tests: ./scripts/test.sh"
fi
if [ "$ENABLE_PYTHON" = "ON" ]; then
    print_info "  Test Python: python -c 'import tensorfuse; print(tensorfuse.__version__)'"
fi
print_info "  Run benchmarks: ./scripts/benchmark.sh" 