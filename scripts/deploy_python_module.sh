#!/bin/bash
set -e

# Deploy Python module script
# This script copies the compiled module from build directory to the Python package location

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Build and source directories
BUILD_MODULE_DIR="$PROJECT_ROOT/build/src/python"
PYTHON_PACKAGE_DIR="$PROJECT_ROOT/src/python/tensorfuse"

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

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info "TensorFuse Python Module Deployment"
print_info "===================================="

# Check if build directory exists
if [ ! -d "$BUILD_MODULE_DIR" ]; then
    print_error "Build directory not found: $BUILD_MODULE_DIR"
    print_error "Please run the build first: ./scripts/build.sh"
    exit 1
fi

# Find the compiled module
MODULE_FILE=$(find "$BUILD_MODULE_DIR" -name "_tensorfuse*.so" | head -1)

if [ -z "$MODULE_FILE" ]; then
    print_error "Compiled module not found in $BUILD_MODULE_DIR"
    print_error "Please ensure the build completed successfully"
    exit 1
fi

print_info "Found compiled module: $MODULE_FILE"

# Create Python package directory if it doesn't exist
if [ ! -d "$PYTHON_PACKAGE_DIR" ]; then
    print_info "Creating Python package directory: $PYTHON_PACKAGE_DIR"
    mkdir -p "$PYTHON_PACKAGE_DIR"
fi

# Remove any existing module files to avoid conflicts
print_info "Cleaning up existing module files..."
rm -f "$PYTHON_PACKAGE_DIR"/_tensorfuse*.so

# Copy the module
MODULE_NAME=$(basename "$MODULE_FILE")
TARGET_PATH="$PYTHON_PACKAGE_DIR/$MODULE_NAME"

print_info "Copying module to: $TARGET_PATH"
cp "$MODULE_FILE" "$TARGET_PATH"

# Verify the copy
if [ -f "$TARGET_PATH" ]; then
    print_info "Module deployment successful!"
    print_info "Module size: $(du -h "$TARGET_PATH" | cut -f1)"
    print_info "Module permissions: $(ls -la "$TARGET_PATH" | cut -d' ' -f1)"
else
    print_error "Module deployment failed!"
    exit 1
fi

# Test basic import
print_info "Testing Python import..."
cd "$PROJECT_ROOT"
python3 -c "
import sys
sys.path.insert(0, 'src/python')
try:
    import tensorfuse
    print('✅ Import successful')
    print(f'Module version: {tensorfuse.__version__}')
except Exception as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)
"

print_info "Python module deployment completed successfully!" 