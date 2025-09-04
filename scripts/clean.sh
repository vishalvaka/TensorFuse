#!/bin/bash
set -e

# TensorFuse Clean Script
# This script removes build artifacts and caches

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Get workspace directory
WORKSPACE_DIR="${WORKSPACE_DIR:-$(pwd)}"
BUILD_DIR="${WORKSPACE_DIR}/build"

print_info "Cleaning TensorFuse build artifacts..."
print_info "Workspace: ${WORKSPACE_DIR}"

cd "${WORKSPACE_DIR}"

# Clean build directory
if [ -d "${BUILD_DIR}" ]; then
    print_info "Removing build directory: ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
else
    print_info "Build directory not found, nothing to clean"
fi

# Clean Python cache
print_info "Cleaning Python cache..."
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true

# Clean CMake cache
print_info "Cleaning CMake cache..."
find . -name "CMakeCache.txt" -delete
find . -name "CMakeFiles" -type d -exec rm -rf {} +

# Clean other artifacts
print_info "Cleaning other artifacts..."
find . -name "*.so" -not -path "./build/*" -delete
find . -name "*.o" -not -path "./build/*" -delete
find . -name "*.a" -not -path "./build/*" -delete

print_info "Clean completed successfully!" 