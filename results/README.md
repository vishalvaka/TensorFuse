# TensorFuse Benchmark Results

This directory contains benchmark results and analysis tools for TensorFuse performance evaluation.

## 📁 Directory Structure

```
results/
├── benchmarks/              # Current benchmark results
│   ├── bert_baseline_results.json
│   └── roofline_data.txt
├── archive/                 # Archived results with timestamps
│   ├── results_20240712_120000/
│   ├── results_20240712_130000/
│   └── ...
└── README.md               # This file
```

## 🚀 Quick Start

### View Results
```bash
# View all benchmark results
make results

# View performance summary only
make results-summary

# View specific components
python3 scripts/view_results.py --bert-only
python3 scripts/view_results.py --roofline-only
```

### Manage Results
```bash
# Organize and copy results from build directory
make results-organize

# Archive current results with timestamp
make results-archive

# Compare with archived results
make results-compare
```

## 📊 Current Performance Status

Based on the latest benchmark results:

- **Current Speedup**: 1.05x (vs cuBLAS baseline)
- **Target Speedup**: 2.0x - 7.0x
- **Status**: ❌ **NEEDS OPTIMIZATION**

### Performance Breakdown

| Component | TensorFuse | cuBLAS | Speedup |
|-----------|------------|--------|---------|
| BERT-base | 1.807ms | 1.904ms | 1.05x |
| Average GEMM | 13.51 TFLOPS | 16.25 TFLOPS | 0.83x |

### Key Findings

1. **Overall Performance**: TensorFuse shows modest improvement (1.05x) on BERT-base
2. **Individual Kernels**: Most kernels are **slower** than cuBLAS (0.83x average)
3. **Optimization Needed**: Significant work required to reach 2x+ target

## 🔍 Analysis Tools

### 1. Results Viewer (`scripts/view_results.py`)

Comprehensive results analysis with multiple views:

```bash
# Full analysis
python3 scripts/view_results.py

# Quick summary
python3 scripts/view_results.py --summary

# Specific analysis
python3 scripts/view_results.py --bert-only
python3 scripts/view_results.py --roofline-only
```

### 2. Results Organizer (`scripts/organize_results.py`)

Manage and compare benchmark results:

```bash
# Copy results from build directory
python3 scripts/organize_results.py --copy

# Archive current results
python3 scripts/organize_results.py --archive

# Compare with historical results
python3 scripts/organize_results.py --compare

# Generate comprehensive report
python3 scripts/organize_results.py --report

# Do everything
python3 scripts/organize_results.py --all
```

## 📈 Performance Optimization Areas

Based on current results, focus optimization on:

1. **Kernel Fusion Efficiency**
   - Current fused kernels are slower than separate operations
   - Need to optimize memory access patterns in fused kernels

2. **Memory Access Patterns**
   - Roofline analysis shows compute-bound behavior
   - Opportunity for memory bandwidth optimization

3. **Tensor Core Utilization**
   - Low compute efficiency (0.1%) suggests poor Tensor Core usage
   - Need to align data layouts for optimal Tensor Core performance

4. **Mixed Precision Optimization**
   - FP16/INT8 paths may not be optimally implemented
   - Focus on precision-specific optimizations

## 🎯 Performance Goals

| Goal | Current | Target | Status |
|------|---------|---------|--------|
| BERT-base Speedup | 1.05x | 2.0x - 7.0x | ❌ Needs Work |
| Tensor Core Efficiency | 0.1% | 50%+ | ❌ Needs Work |
| Memory Efficiency | 24% | 80%+ | ❌ Needs Work |

## 💡 Usage Tips

1. **Run benchmarks regularly** to track progress:
   ```bash
   ./scripts/benchmark.sh
   make results-organize
   ```

2. **Archive before major changes**:
   ```bash
   make results-archive
   ```

3. **Compare after optimization**:
   ```bash
   make results-compare
   ```

4. **Focus on bottlenecks** identified in roofline analysis

## 📝 Result File Formats

### BERT Baseline Results (`bert_baseline_results.json`)
```json
{
  "bert_base_benchmark": {
    "config": { "batch_size": 32, "seq_len": 128, ... },
    "results": [
      {
        "name": "cuBLASLt_baseline",
        "avg_time_ms": 1.904,
        "tflops": 20.31,
        "memory_bandwidth_gbps": 49.57
      },
      {
        "name": "TensorFuse_fused",
        "avg_time_ms": 1.807,
        "tflops": 21.41,
        "memory_bandwidth_gbps": 52.24
      }
    ]
  }
}
```

### Roofline Data (`roofline_data.txt`)
```
# Roofline Plot Data
# Operational_Intensity(FLOPS/byte) Performance(TFLOPS) Kernel_Name
36.5612 1.18814 TensorFuse_128x512x256
36.5612 3.72364 cuBLAS_128x512x256
...
```

## 🔧 Troubleshooting

### No Results Found
```bash
# Run benchmarks first
./scripts/benchmark.sh

# Or copy existing results
make results-organize
```

### Old Results
```bash
# Archive current results
make results-archive

# Run fresh benchmarks
make clean && make build
./scripts/benchmark.sh
```

### Performance Regression
```bash
# Compare with archived results
make results-compare

# Identify which optimization caused regression
# Revert changes and re-benchmark
```

---

📚 **For more details**: See the main project documentation and benchmark scripts in `scripts/` directory. 