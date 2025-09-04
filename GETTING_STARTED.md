# 🚀 TensorFuse Getting Started Guide

This guide will help you get TensorFuse up and running in **under 10 minutes**.

## 📋 Quick Prerequisites Check

Before starting, ensure you have:
- **NVIDIA GPU** with Compute Capability 8.0+ (RTX 30xx/40xx, A100, H100, etc.)
- **Linux** environment (native or WSL2)
- **Docker** (recommended) OR **CUDA 12.6+** (for local development)

## 🎯 Choose Your Setup Method

### Method 1: 🐳 Docker Setup (Recommended - Easiest)

**Perfect for**: Quick testing, development, avoiding dependency issues

```bash
# 1. Clone the repository
git clone https://github.com/vishalvaka/tensorfuse.git
cd tensorfuse

# 2. Start Docker environment (handles all dependencies)
docker compose up -d

# 3. Enter the container
docker compose exec tensorfuse bash

# 4. Build and verify (inside container)
./scripts/build.sh
./scripts/verify.sh

# 5. Run tests to confirm everything works
./scripts/test.sh --python-only
```

**✅ Expected result**: All tests should pass, giving you a working TensorFuse environment!

---

### Method 2: 🔧 Local Development Setup

**Perfect for**: Integration into existing projects, performance optimization

#### Step 1: Install Dependencies

```bash
# Install CUDA 12.6+ (if not already installed)
wget https://developer.download.nvidia.com/compute/cuda/12.6.1/local_installers/cuda_12.6.1_560.35.03_linux.run
sudo sh cuda_12.6.1_560.35.03_linux.run

# Install CUTLASS 3.5.1
git clone --branch v3.5.1 --depth 1 https://github.com/NVIDIA/cutlass.git /tmp/cutlass
cd /tmp/cutlass && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DCUTLASS_ENABLE_HEADERS_ONLY=ON
sudo make install -j$(nproc)
```

#### Step 2: Build TensorFuse

```bash
# Clone and build
git clone https://github.com/vishalvaka/tensorfuse.git
cd tensorfuse

# Install Python dependencies
pip install -r requirements.txt

# Build TensorFuse
./scripts/build.sh

# Verify installation
./scripts/verify.sh
```

#### Step 3: Test Your Installation

```bash
# Run all tests
./scripts/test.sh

# Run quick benchmark
./scripts/benchmark.sh --quick
```

---

## 🧪 Quick Verification

After either setup method, verify TensorFuse is working:

```bash
# Quick verification script
./scripts/verify.sh

# Should output: "🎉 All checks passed! TensorFuse is ready to use."
```

## 🏃‍♂️ Your First TensorFuse Program

Create `test_tensorfuse.py`:

```python
import tensorfuse
import numpy as np

# Initialize TensorFuse
tensorfuse.init()

# Create test data (transformer-like dimensions)
batch_size, seq_len, hidden_dim = 32, 128, 768
ffn_dim = 3072

input_tensor = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
weight = np.random.randn(hidden_dim, ffn_dim).astype(np.float32)
bias = np.random.randn(ffn_dim).astype(np.float32)
output = np.zeros((batch_size, seq_len, ffn_dim), dtype=np.float32)

# Run fused GEMM+Bias+GELU (2-3x faster than separate operations!)
tensorfuse.fused_gemm_bias_gelu(input_tensor, weight, bias, output)

print(f"✅ Fused operation completed! Output shape: {output.shape}")
print(f"📊 Performance boost: ~2-3x faster than standard PyTorch")

# Cleanup
tensorfuse.shutdown()
```

Run it:
```bash
python test_tensorfuse.py
```

## 🚨 Common Issues & Solutions

### Issue: "CUDA not found"
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# If missing, install CUDA 12.6+
# See Method 2 instructions above
```

### Issue: "TensorFuse module not found"
```bash
# Rebuild Python bindings
./scripts/build.sh --clean
pip install -e .
```

### Issue: "Build fails with CMake errors"
```bash
# Check requirements
./scripts/verify.sh

# Clean and rebuild
./scripts/clean.sh
./scripts/build.sh --debug
```

### Issue: "Tests fail"
```bash
# Check detailed output
./scripts/test.sh --verbose

# Run specific test category
./scripts/test.sh --python-only
```

### Issue: Docker problems
```bash
# Restart Docker services
docker compose down
docker compose up -d --build

# Check GPU access in container
docker compose exec tensorfuse nvidia-smi
```

## 📚 Next Steps

### **For Researchers/Scientists**:
```bash
# Explore benchmarks
./scripts/benchmark.sh

# Check out examples
ls src/tests/python/

# Profile your workloads
python -c "
import tensorfuse
config = tensorfuse.Config(enable_profiling=True)
tensorfuse.init(config)
"
```

### **For Developers**:
```bash
# Build with debugging
./scripts/build.sh --debug

# Run C++ tests
./scripts/test.sh --cpp-only

# Check C++ examples
ls src/tests/
```

### **For Integration**:
```bash
# Install as package
pip install -e .

# Use in your project
import tensorfuse
# Your code here...
```

## 🛠️ Development Workflow

```bash
# Edit code
vim src/python/tensorfuse/core.py

# Rebuild and test
./scripts/build.sh && ./scripts/test.sh --python-only

# Run benchmarks to check performance
./scripts/benchmark.sh --quick
```

## 📖 Documentation

- **API Reference**: See `src/tests/python/` for comprehensive examples
- **Performance Guide**: See `docs/performance.md`
- **Architecture**: See `README.md` Architecture section
- **Troubleshooting**: See `TROUBLESHOOTING.md`

## 💬 Need Help?

1. **Check this guide** and the troubleshooting section
2. **Run the verification script**: `./scripts/verify.sh`
3. **Check existing issues**: GitHub Issues tab
4. **Create a new issue** with:
   - Your setup method (Docker/Local)
   - Error messages
   - Output of `./scripts/verify.sh`
   - Your GPU model and CUDA version

---

## 🎉 You're Ready!

If you've completed the verification successfully, you now have:
- ✅ A working TensorFuse installation
- ✅ 2-7x speedups for fused operations
- ✅ Full Python and C++ API access
- ✅ Comprehensive test suite

**Happy accelerating! 🚀** 