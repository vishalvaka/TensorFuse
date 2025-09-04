#!/bin/bash

# ==============================================================================
# TensorFuse Nsight Systems Profiling Script
# ==============================================================================

set -e

# Configuration
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${WORKSPACE_DIR}/build"
RESULTS_DIR="${WORKSPACE_DIR}/benchmarks/results"
NSIGHT_OUTPUT_DIR="${RESULTS_DIR}/nsight_traces"

# Default parameters
PROFILE_DURATION=30  # seconds
PROFILE_DELAY=2      # seconds
CAPTURE_CUDA=true
CAPTURE_CUBLAS=true
CAPTURE_NVTX=true
CAPTURE_OSRT=true
VERBOSE=false

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
    echo -e "${BLUE}$1${NC}"
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --duration SECONDS    Profile duration (default: 30)"
    echo "  -o, --output DIR          Output directory (default: benchmarks/results/nsight_traces)"
    echo "  --no-cuda                 Disable CUDA API tracing"
    echo "  --no-cublas               Disable cuBLAS tracing"
    echo "  --no-nvtx                 Disable NVTX tracing"
    echo "  --no-osrt                 Disable OS runtime tracing"
    echo "  -v, --verbose             Enable verbose output"
    echo "  -h, --help                Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--duration)
            PROFILE_DURATION="$2"
            shift 2
            ;;
        -o|--output)
            NSIGHT_OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-cuda)
            CAPTURE_CUDA=false
            shift
            ;;
        --no-cublas)
            CAPTURE_CUBLAS=false
            shift
            ;;
        --no-nvtx)
            CAPTURE_NVTX=false
            shift
            ;;
        --no-osrt)
            CAPTURE_OSRT=false
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

print_section "🚀 TensorFuse Nsight Systems Profiling"
print_section "======================================="

# Check prerequisites
print_info "Checking prerequisites..."

# Check if nsys is available
if ! command -v nsys &> /dev/null; then
    print_error "nsys (Nsight Systems) is not installed or not in PATH"
    print_error "Please install Nsight Systems: https://developer.nvidia.com/nsight-systems"
    exit 1
fi

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. CUDA profiling requires NVIDIA GPU."
    exit 1
fi

# Check if benchmarks are built
if [ ! -f "${BUILD_DIR}/benchmarks/bert_baseline" ]; then
    print_error "BERT baseline benchmark not found. Please build first:"
    print_error "  cd ${WORKSPACE_DIR} && ./scripts/build.sh"
    exit 1
fi

# Create output directory
mkdir -p "${NSIGHT_OUTPUT_DIR}"

# Get GPU information
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | head -1)
print_info "GPU: ${GPU_INFO}"

# Build nsys command
NSYS_CMD="nsys profile"
NSYS_CMD+=" --duration=${PROFILE_DURATION}"
NSYS_CMD+=" --delay=${PROFILE_DELAY}"
NSYS_CMD+=" --stats=true"
NSYS_CMD+=" --force-overwrite=true"

# Add tracing options
TRACE_OPTIONS=""
if [ "$CAPTURE_CUDA" = true ]; then
    TRACE_OPTIONS+=",cuda"
fi
if [ "$CAPTURE_CUBLAS" = true ]; then
    TRACE_OPTIONS+=",cublas"
fi
if [ "$CAPTURE_NVTX" = true ]; then
    TRACE_OPTIONS+=",nvtx"
fi
if [ "$CAPTURE_OSRT" = true ]; then
    TRACE_OPTIONS+=",osrt"
fi

# Remove leading comma
TRACE_OPTIONS="${TRACE_OPTIONS#,}"

if [ -n "$TRACE_OPTIONS" ]; then
    NSYS_CMD+=" --trace=${TRACE_OPTIONS}"
fi

if [ "$VERBOSE" = true ]; then
    NSYS_CMD+=" --verbose"
fi

print_section "🔍 Running BERT-base Baseline Profiling"
print_section "========================================"

# Profile BERT baseline benchmark
BERT_TRACE_FILE="${NSIGHT_OUTPUT_DIR}/bert_baseline_trace.nsys-rep"
BERT_CMD="${BUILD_DIR}/benchmarks/bert_baseline"

print_info "Profiling BERT baseline benchmark..."
print_info "Command: ${NSYS_CMD} --output=${BERT_TRACE_FILE} ${BERT_CMD}"

cd "${WORKSPACE_DIR}"
${NSYS_CMD} --output="${BERT_TRACE_FILE}" ${BERT_CMD}

if [ $? -eq 0 ]; then
    print_info "✅ BERT baseline profiling completed successfully"
else
    print_error "❌ BERT baseline profiling failed"
    exit 1
fi

print_section "🔍 Running Roofline Analysis Profiling"
print_section "======================================="

# Profile roofline analysis
if [ -f "${BUILD_DIR}/benchmarks/roofline_analysis" ]; then
    ROOFLINE_TRACE_FILE="${NSIGHT_OUTPUT_DIR}/roofline_analysis_trace.nsys-rep"
    ROOFLINE_CMD="${BUILD_DIR}/benchmarks/roofline_analysis"
    
    print_info "Profiling roofline analysis..."
    print_info "Command: ${NSYS_CMD} --output=${ROOFLINE_TRACE_FILE} ${ROOFLINE_CMD}"
    
    ${NSYS_CMD} --output="${ROOFLINE_TRACE_FILE}" ${ROOFLINE_CMD}
    
    if [ $? -eq 0 ]; then
        print_info "✅ Roofline analysis profiling completed successfully"
    else
        print_warn "⚠️  Roofline analysis profiling failed (optional)"
    fi
else
    print_warn "⚠️  Roofline analysis benchmark not found (optional)"
fi

print_section "🔍 Running Individual Kernel Profiling"
print_section "======================================="

# Profile individual kernels with different matrix sizes
KERNEL_SIZES=(
    "128,512,256"
    "512,1024,512"
    "1024,2048,1024"
    "2048,4096,2048"
)

for size in "${KERNEL_SIZES[@]}"; do
    IFS=',' read -r M N K <<< "$size"
    
    print_info "Profiling kernel with size ${M}x${N}x${K}..."
    
    # Create a temporary program for this specific kernel size
    KERNEL_TRACE_FILE="${NSIGHT_OUTPUT_DIR}/kernel_${M}x${N}x${K}_trace.nsys-rep"
    
    # Use Python to run specific kernel size
    KERNEL_CMD="python -c \"
import sys
sys.path.insert(0, 'src/python')
import tensorfuse
import numpy as np
import time

# Initialize TensorFuse
tensorfuse.init()

# Create test matrices
M, N, K = $M, $N, $K
print(f'Running kernel with size {M}x{N}x{K}...')

A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)
bias = np.random.randn(N).astype(np.float32)
output = np.zeros((M, N), dtype=np.float32)

# Warmup
for i in range(5):
    tensorfuse.fused_gemm_bias_gelu(A, B, bias, output)

# Run for profiling
start_time = time.time()
for i in range(100):
    tensorfuse.fused_gemm_bias_gelu(A, B, bias, output)
end_time = time.time()

print(f'Completed 100 iterations in {end_time - start_time:.3f} seconds')
print(f'Average time per iteration: {(end_time - start_time) * 10:.3f} ms')

tensorfuse.shutdown()
\""
    
    # Run with reduced duration for individual kernels
    KERNEL_NSYS_CMD="${NSYS_CMD/duration=${PROFILE_DURATION}/duration=15}"
    
    ${KERNEL_NSYS_CMD} --output="${KERNEL_TRACE_FILE}" bash -c "${KERNEL_CMD}"
    
    if [ $? -eq 0 ]; then
        print_info "✅ Kernel ${M}x${N}x${K} profiling completed"
    else
        print_warn "⚠️  Kernel ${M}x${N}x${K} profiling failed"
    fi
done

print_section "📊 Generating Reports"
print_section "====================="

# Generate summary reports from traces
print_info "Generating summary reports..."

for trace_file in "${NSIGHT_OUTPUT_DIR}"/*.nsys-rep; do
    if [ -f "$trace_file" ]; then
        base_name=$(basename "$trace_file" .nsys-rep)
        report_file="${NSIGHT_OUTPUT_DIR}/${base_name}_summary.txt"
        
        print_info "Generating report for $(basename "$trace_file")..."
        
        # Generate summary report
        nsys stats --report gpukernsum,gputrace "$trace_file" > "$report_file" 2>&1
        
        if [ $? -eq 0 ]; then
            print_info "✅ Report generated: $report_file"
        else
            print_warn "⚠️  Failed to generate report for $trace_file"
        fi
    fi
done

print_section "📈 Performance Analysis"
print_section "======================="

# Create a comprehensive performance report
PERFORMANCE_REPORT="${NSIGHT_OUTPUT_DIR}/performance_analysis.md"

cat > "$PERFORMANCE_REPORT" << EOF
# TensorFuse Performance Analysis

## System Information
- **GPU**: ${GPU_INFO}
- **Profile Date**: $(date)
- **Profile Duration**: ${PROFILE_DURATION} seconds
- **Trace Options**: ${TRACE_OPTIONS}

## Files Generated
EOF

# List all generated files
for file in "${NSIGHT_OUTPUT_DIR}"/*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        filesize=$(du -h "$file" | cut -f1)
        echo "- **${filename}** (${filesize})" >> "$PERFORMANCE_REPORT"
    fi
done

cat >> "$PERFORMANCE_REPORT" << EOF

## Analysis Instructions

### 1. Open in Nsight Systems GUI
To analyze the traces visually:
\`\`\`bash
nsys-ui ${NSIGHT_OUTPUT_DIR}/bert_baseline_trace.nsys-rep
\`\`\`

### 2. Key Metrics to Look For
- **GPU Utilization**: Should be >80% for optimal performance
- **Memory Bandwidth**: Compare against theoretical peak
- **Kernel Launch Overhead**: Minimize gaps between kernels
- **Tensor Core Utilization**: Check for mixed precision usage

### 3. Command Line Analysis
Generate detailed reports:
\`\`\`bash
# GPU kernel summary
nsys stats --report gpukernsum ${NSIGHT_OUTPUT_DIR}/bert_baseline_trace.nsys-rep

# CUDA API trace
nsys stats --report cudaapisum ${NSIGHT_OUTPUT_DIR}/bert_baseline_trace.nsys-rep

# Memory operations
nsys stats --report memop ${NSIGHT_OUTPUT_DIR}/bert_baseline_trace.nsys-rep
\`\`\`

### 4. Roofline Analysis
The roofline data can be plotted using:
\`\`\`bash
# If gnuplot is available
gnuplot -e "
set logscale x;
set xlabel 'Operational Intensity (FLOPS/byte)';
set ylabel 'Performance (TFLOPS)';
plot '${RESULTS_DIR}/roofline_data.txt' using 1:2 with points title 'TensorFuse Kernels'
"
\`\`\`

## Next Steps
1. Identify bottlenecks in the Nsight Systems GUI
2. Focus optimization efforts on memory-bound vs compute-bound kernels
3. Optimize kernel launch patterns and memory access
4. Validate improvements with follow-up profiling runs
EOF

print_section "🎉 Profiling Complete!"
print_section "====================="

print_info "✅ All profiling sessions completed successfully"
print_info "📁 Results saved to: ${NSIGHT_OUTPUT_DIR}"
print_info "📊 Performance report: ${PERFORMANCE_REPORT}"
print_info ""
print_info "🔍 To analyze results:"
print_info "  1. Open Nsight Systems GUI: nsys-ui ${NSIGHT_OUTPUT_DIR}/bert_baseline_trace.nsys-rep"
print_info "  2. Read analysis report: cat ${PERFORMANCE_REPORT}"
print_info "  3. Generate additional reports: nsys stats --help"
print_info ""
print_info "💡 Focus areas for optimization:"
print_info "  - GPU utilization >80%"
print_info "  - Memory bandwidth efficiency"
print_info "  - Kernel launch overhead"
print_info "  - Tensor Core utilization"

# Create a simple script to open the main trace
cat > "${NSIGHT_OUTPUT_DIR}/open_trace.sh" << EOF
#!/bin/bash
echo "Opening main BERT baseline trace in Nsight Systems..."
nsys-ui "${NSIGHT_OUTPUT_DIR}/bert_baseline_trace.nsys-rep"
EOF

chmod +x "${NSIGHT_OUTPUT_DIR}/open_trace.sh"

print_info "🚀 Quick start: ${NSIGHT_OUTPUT_DIR}/open_trace.sh" 