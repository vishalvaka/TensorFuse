# Docker Development Guide

This guide covers Docker-based development for TensorFuse, providing a consistent environment across different systems.

## Prerequisites

### 1. Install Docker
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y docker.io docker-compose

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (logout/login required)
sudo usermod -aG docker $USER
```

### 2. Install NVIDIA Container Toolkit
```bash
# Add NVIDIA Container Toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the toolkit
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 3. Verify GPU Access
```bash
# Test GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.6-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### 1. Launch Development Environment
```bash
# Clone repository
git clone https://github.com/vishalvaka/tensorfuse.git
cd tensorfuse

# Start the development environment
docker compose up -d

# Access the container
docker compose exec tensorfuse bash
```

### 2. Build and Test
```bash
# Inside the container
./scripts/build.sh
./scripts/test.sh
```

## Development Workflow

### Container Management
```bash
# Start environment
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f tensorfuse

# Access container
docker compose exec tensorfuse bash

# Stop environment
docker compose down

# Rebuild image
docker compose build --no-cache
```

### Build Commands
```bash
# Inside container
./scripts/build.sh        # Full build
./scripts/build.sh --clean # Clean build
./scripts/clean.sh         # Clean only
```

### Testing Commands
```bash
# Run all tests
./scripts/test.sh

# Run specific test suites
./scripts/test-cpp.sh      # C++ tests only
./scripts/test-python.sh   # Python tests only

# Run individual tests
./build/src/tests/test_kernels
python -m pytest src/tests/python/test_core_bindings.py -v
```

## Docker Environment Features

### Pre-installed Tools
- **CUDA 12.6**: Latest CUDA toolkit
- **CUTLASS 3.5.1**: High-performance CUDA templates
- **Python 3.10**: With all required packages
- **CMake 3.27**: Modern build system
- **Ninja**: Fast build tool
- **GDB**: Debugging tools
- **Nsight Systems/Compute**: Profiling tools

### Volume Mounts
- **Source Code**: `/workspace` (read-write)
- **Build Cache**: `/workspace/build` (persistent)
- **Package Cache**: `/workspace/cache` (persistent)

### Environment Variables
- `CUDA_HOME=/usr/local/cuda-12.6`
- `TENSORFUSE_CUDA_ARCHITECTURES=89;90`
- `TENSORFUSE_BUILD_TYPE=Release`
- `TENSORFUSE_ENABLE_PROFILING=ON`

### Exposed Ports
- **8888**: Jupyter Lab (if enabled)
- **8889-8890**: Additional ports for services

## Build Scripts

### `/scripts/build.sh`
```bash
#!/bin/bash
set -e

# Build TensorFuse library
echo "Building TensorFuse..."
cd /workspace

# Create build directory
mkdir -p build && cd build

# Configure CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DTENSORFUSE_ENABLE_PROFILING=OFF \
    -DTENSORFUSE_BUILD_TESTS=ON \
    -DTENSORFUSE_BUILD_PYTHON=ON

# Build
make -j$(nproc)

echo "Build completed successfully!"
```

### `/scripts/test.sh`
```bash
#!/bin/bash
set -e

echo "Running TensorFuse tests..."
cd /workspace

# Run C++ tests
echo "Running C++ tests..."
cd build && ctest --verbose --output-on-failure

# Run Python tests
echo "Running Python tests..."
cd /workspace/src/tests/python
python -m pytest -v --tb=short

echo "All tests completed!"
```

## Development Tips

### Code Editing
- Edit code on your host machine using your favorite IDE
- Files are automatically synced to the container via volume mounts
- No need to restart the container for code changes

### Debugging
```bash
# Debug with GDB
docker compose exec tensorfuse gdb ./build/src/tests/test_kernels

# Check GPU utilization
docker compose exec tensorfuse nvidia-smi

# Monitor memory usage
docker compose exec tensorfuse watch -n 1 nvidia-smi
```

### Performance Profiling
```bash
# Profile with Nsight Systems
docker compose exec tensorfuse nsys profile --output=profile.nsys-rep ./build/src/tests/test_kernels

# Profile with Nsight Compute
docker compose exec tensorfuse ncu --output=profile.ncu-rep ./build/src/tests/test_kernels
```

## Troubleshooting

### Common Issues

1. **GPU not accessible**
   ```bash
   # Check NVIDIA Container Toolkit installation
   docker run --rm --gpus all nvidia/cuda:12.6-base nvidia-smi
   ```

2. **Permission denied**
   ```bash
   # Add user to docker group
   sudo usermod -aG docker $USER
   # Logout and login again
   ```

3. **Build failures**
   ```bash
   # Clean build
   docker compose exec tensorfuse ./scripts/clean.sh
   docker compose exec tensorfuse ./scripts/build.sh --clean
   ```

4. **Out of memory**
   ```bash
   # Reduce parallel jobs
   docker compose exec tensorfuse make -j2  # instead of -j$(nproc)
   ```

### Debug Commands
```bash
# Check CUDA installation
docker compose exec tensorfuse nvcc --version

# Check CUTLASS installation
docker compose exec tensorfuse find /usr/local -name "*cutlass*" -type d

# Check Python environment
docker compose exec tensorfuse python -c "import torch; print(torch.cuda.is_available())"

# Check build configuration
docker compose exec tensorfuse cmake -L build/
```

## Container Customization

### Environment Variables
```yaml
# In docker-compose.yml
environment:
  - TENSORFUSE_CUDA_ARCHITECTURES=89;90
  - TENSORFUSE_BUILD_TYPE=Release
  - TENSORFUSE_ENABLE_PROFILING=ON
```

### Build Arguments
```yaml
# In docker-compose.yml
build:
  context: .
  dockerfile: docker/cuda12.6-runtime.Dockerfile
  args:
    CUDA_VERSION: 12.6
    CUTLASS_VERSION: 3.5.1
```

### Additional Packages
```dockerfile
# Add to Dockerfile
RUN apt-get update && apt-get install -y \
    your-additional-package \
    && rm -rf /var/lib/apt/lists/*
```

## Best Practices

1. **Use Docker Compose**: Easier management than direct Docker commands
2. **Volume Mounts**: Keep source code on host for IDE integration
3. **Persistent Containers**: Use `docker compose up -d` for long-running development
4. **Clean Builds**: Use `./scripts/clean.sh` when switching branches
5. **Resource Limits**: Monitor GPU memory usage during development

## Integration with IDEs

### VS Code
1. Install "Remote - Containers" extension
2. Use "Remote-Containers: Open Folder in Container"
3. Configure `.devcontainer/devcontainer.json`

### PyCharm
1. Configure Docker interpreter
2. Set remote interpreter to container Python
3. Map source directories

### vim/emacs
1. Edit files on host machine
2. Build/test in container
3. Use terminal multiplexer (tmux) in container 