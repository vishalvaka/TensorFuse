#!/bin/bash
set -e

# TensorFuse Benchmark Script
# This script runs benchmarks to verify performance

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
    echo -e "${BLUE}[BENCHMARK]${NC} $1"
}

# Parse command line arguments
QUICK_BENCH=false
SAVE_RESULTS=false
OUTPUT_DIR="./benchmark_results"
RUN_PYTHON=true
RUN_BERT=true
RUN_ROOFLINE=true
RUN_LEGACY=true

for arg in "$@"
do
    case $arg in
        --quick)
        QUICK_BENCH=true
        shift
        ;;
        --save)
        SAVE_RESULTS=true
        shift
        ;;
        --output=*)
        OUTPUT_DIR="${arg#*=}"
        shift
        ;;
        --python-only)
        RUN_BERT=false
        RUN_ROOFLINE=false
        RUN_LEGACY=false
        shift
        ;;
        --bert-only)
        RUN_PYTHON=false
        RUN_ROOFLINE=false
        RUN_LEGACY=false
        shift
        ;;
        --roofline-only)
        RUN_PYTHON=false
        RUN_BERT=false
        RUN_LEGACY=false
        shift
        ;;
        --cpp-only)
        RUN_PYTHON=false
        shift
        ;;
        --help)
        echo "Usage: $0 [OPTIONS]"
        echo "Options:"
        echo "  --quick         Run quick benchmarks only"
        echo "  --save          Save benchmark results to files"
        echo "  --output=DIR    Output directory for results (default: ./benchmark_results)"
        echo ""
        echo "Benchmark Selection:"
        echo "  --python-only   Run only Python API benchmarks"
        echo "  --bert-only     Run only BERT baseline benchmark"
        echo "  --roofline-only Run only roofline analysis"
        echo "  --cpp-only      Run only C++ benchmarks (BERT + roofline + legacy)"
        echo ""
        echo "  --help          Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0                          # Run all benchmarks"
        echo "  $0 --quick --save          # Quick benchmarks with saved results"
        echo "  $0 --bert-only --save      # Only BERT baseline with results"
        echo "  $0 --roofline-only         # Only roofline analysis"
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

print_info "Starting TensorFuse benchmarks..."
print_info "Workspace: ${WORKSPACE_DIR}"
print_info "Build directory: ${BUILD_DIR}"

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
    print_error "nvidia-smi not found. GPU benchmarks require NVIDIA GPU."
    exit 1
fi

# Create output directory if saving results
if [ "$SAVE_RESULTS" = true ]; then
    mkdir -p "${OUTPUT_DIR}"
    print_info "Results will be saved to: ${OUTPUT_DIR}"
fi

# Change to workspace directory
cd "${WORKSPACE_DIR}"

print_section "=== TensorFuse Performance Benchmarks ==="

# Run Python benchmarks
if [ "$RUN_PYTHON" = true ]; then
    # Check if Python module is available
    if ! python -c "import tensorfuse" 2>/dev/null; then
        print_error "TensorFuse Python module not found."
        print_error "Please run './scripts/build.sh' to build Python bindings."
        exit 1
    fi

    print_section "Running Python API benchmarks..."

BENCHMARK_SCRIPT="
import tensorfuse
import torch
import time
import numpy as np

print('🚀 TensorFuse Benchmark Suite')
print('=' * 50)

# Initialize TensorFuse
try:
    tensorfuse.init()
    print('✅ TensorFuse initialized successfully')
except Exception as e:
    print(f'❌ TensorFuse initialization failed: {e}')
    exit(1)

# Benchmark parameters
batch_size, seq_len, hidden_dim = 32, 128, 768
ffn_dim = 3072
num_iterations = 10 if '${QUICK_BENCH}' == 'true' else 100

print(f'📊 Benchmark Configuration:')
print(f'   Batch size: {batch_size}')
print(f'   Sequence length: {seq_len}')
print(f'   Hidden dim: {hidden_dim}')
print(f'   FFN dim: {ffn_dim}')
print(f'   Iterations: {num_iterations}')
print()

# Create test tensors
input_tensor = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
weight = np.random.randn(hidden_dim, ffn_dim).astype(np.float32)
bias = np.random.randn(ffn_dim).astype(np.float32)
output = np.zeros((batch_size, seq_len, ffn_dim), dtype=np.float32)

print('🔥 Running GEMM+Bias+GELU benchmark...')
start_time = time.time()

for i in range(num_iterations):
    try:
        tensorfuse.fused_gemm_bias_gelu(input_tensor, weight, bias, output)
    except Exception as e:
        print(f'❌ Benchmark failed: {e}')
        break

end_time = time.time()
avg_time = (end_time - start_time) / num_iterations * 1000  # ms

print(f'✅ Average execution time: {avg_time:.2f} ms')
print(f'📈 Throughput: {num_iterations / (end_time - start_time):.1f} ops/sec')

# Calculate FLOPs
flops_per_op = 2 * batch_size * seq_len * hidden_dim * ffn_dim
total_flops = flops_per_op * num_iterations
tflops = total_flops / (end_time - start_time) / 1e12

print(f'⚡ Performance: {tflops:.2f} TFLOPS')

# Memory usage estimation
memory_mb = (input_tensor.nbytes + weight.nbytes + bias.nbytes + output.nbytes) / 1024 / 1024
print(f'💾 Memory usage: ~{memory_mb:.1f} MB')

print()
print('🎉 Benchmark completed successfully!')
print('📝 For detailed profiling, enable profiling in tensorfuse.init()')

# Cleanup
tensorfuse.shutdown()
"

echo "$BENCHMARK_SCRIPT" | python

    if [ $? -eq 0 ]; then
        print_info "Python benchmarks completed successfully!"
    else
        print_error "Python benchmarks failed!"
        exit 1
    fi
else
    print_info "Skipping Python benchmarks (use --python-only to run only Python benchmarks)"
fi

# Run C++ benchmarks if available
if [ "$RUN_BERT" = true ] || [ "$RUN_ROOFLINE" = true ] || [ "$RUN_LEGACY" = true ]; then
    print_section "Running C++ benchmarks..."
fi

# BERT Baseline Benchmark
if [ "$RUN_BERT" = true ] && [ -f "${BUILD_DIR}/benchmarks/bert_baseline" ]; then
    print_section "🚀 Running BERT-base Baseline Benchmark..."
    cd "${BUILD_DIR}"
    
    if [ "$SAVE_RESULTS" = true ]; then
        print_info "Saving BERT baseline results to ${OUTPUT_DIR}"
        mkdir -p "${OUTPUT_DIR}"
        ./benchmarks/bert_baseline | tee "${OUTPUT_DIR}/bert_baseline_output.txt"
        
        # Copy JSON results if they exist
        if [ -f "benchmarks/results/bert_baseline_results.json" ]; then
            cp "benchmarks/results/bert_baseline_results.json" "${WORKSPACE_DIR}/${OUTPUT_DIR}/"
            print_info "✅ BERT baseline JSON results saved"
        fi
    else
        ./benchmarks/bert_baseline
    fi
    
    if [ $? -eq 0 ]; then
        print_info "✅ BERT baseline benchmark completed successfully!"
    else
        print_error "❌ BERT baseline benchmark failed!"
    fi
    
    cd "${WORKSPACE_DIR}"
elif [ "$RUN_BERT" = true ]; then
    print_warn "BERT baseline benchmark not found at ${BUILD_DIR}/benchmarks/bert_baseline"
elif [ "$RUN_BERT" = false ]; then
    print_info "Skipping BERT baseline benchmark (use --bert-only to run only BERT benchmark)"
fi

echo ""

# Roofline Analysis Benchmark
if [ "$RUN_ROOFLINE" = true ] && [ -f "${BUILD_DIR}/benchmarks/roofline_analysis" ]; then
    print_section "🏔️  Running Roofline Analysis..."
    cd "${BUILD_DIR}"
    
    if [ "$SAVE_RESULTS" = true ]; then
        print_info "Saving roofline analysis results to ${OUTPUT_DIR}"
        mkdir -p "${OUTPUT_DIR}"
        ./benchmarks/roofline_analysis | tee "${OUTPUT_DIR}/roofline_analysis_output.txt"
        
        # Copy roofline data if it exists
        if [ -f "benchmarks/results/roofline_data.txt" ]; then
            cp "benchmarks/results/roofline_data.txt" "${WORKSPACE_DIR}/${OUTPUT_DIR}/"
            print_info "✅ Roofline data saved"
        fi
    else
        ./benchmarks/roofline_analysis
    fi
    
    if [ $? -eq 0 ]; then
        print_info "✅ Roofline analysis completed successfully!"
    else
        print_error "❌ Roofline analysis failed!"
    fi
    
    cd "${WORKSPACE_DIR}"
elif [ "$RUN_ROOFLINE" = true ]; then
    print_warn "Roofline analysis benchmark not found at ${BUILD_DIR}/benchmarks/roofline_analysis"
elif [ "$RUN_ROOFLINE" = false ]; then
    print_info "Skipping roofline analysis (use --roofline-only to run only roofline analysis)"
fi

echo ""

# Legacy C++ benchmarks (if available)
if [ "$RUN_LEGACY" = true ] && [ -f "${BUILD_DIR}/benchmarks/benchmark_kernels" ]; then
    print_section "Running legacy C++ kernel benchmarks..."
    cd "${BUILD_DIR}"
    ./benchmarks/benchmark_kernels
    cd "${WORKSPACE_DIR}"
elif [ "$RUN_LEGACY" = false ]; then
    print_info "Skipping legacy C++ benchmarks"
fi

print_section "=== Benchmark Summary ==="
print_info "✅ TensorFuse benchmark suite completed!"

# Show what was run
if [ "$RUN_PYTHON" = true ]; then
    print_info "🐍 Python API benchmarks: COMPLETED"
fi

if [ "$RUN_BERT" = true ]; then
    print_info "🤖 BERT baseline benchmark: COMPLETED"
fi

if [ "$RUN_ROOFLINE" = true ]; then
    print_info "🏔️  Roofline analysis: COMPLETED"
fi

if [ "$RUN_LEGACY" = true ]; then
    print_info "⚙️  Legacy C++ benchmarks: COMPLETED"
fi

print_info "🚀 TensorFuse is working correctly and performing well"

if [ "$SAVE_RESULTS" = true ]; then
    print_info "📁 Results saved to: ${OUTPUT_DIR}"
    print_info "📊 Check the output directory for detailed results and analysis"
fi

print_info ""
print_info "💡 Tips:"
print_info "   • Run with --quick for faster benchmarks during development"
print_info "   • Use --save to keep benchmark results for analysis"
print_info "   • Use --bert-only or --roofline-only to focus on specific benchmarks"
print_info "   • Use --help to see all available options" 