# TensorFuse Quick Start Guide

This guide will get you up and running with TensorFuse in under 5 minutes using Docker.

## Prerequisites

- NVIDIA GPU with Compute Capability 8.0+ (RTX 30xx, RTX 40xx, A100, H100, etc.)
- Docker and Docker Compose installed
- NVIDIA Container Toolkit installed

## Installation Check

First, verify your system meets the requirements:

```bash
# Check GPU
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.6-base nvidia-smi

# Check Docker Compose
docker compose version
```

## Quick Start

### 1. Get TensorFuse

```bash
git clone https://github.com/vishalvaka/tensorfuse.git
cd tensorfuse
```

### 2. Start Development Environment

```bash
# Start the container
docker compose up -d

# Access the container
docker compose exec tensorfuse bash
```

### 3. Build and Test

```bash
# Inside the container
./scripts/build.sh
./scripts/test.sh
```

That's it! You now have a working TensorFuse installation.

## Next Steps

### Try the Python API

```bash
# Inside the container
python3 -c "
import tensorfuse
print('TensorFuse is working!')
print('Version:', tensorfuse.__version__)
"
```

### Run Performance Tests

```bash
# Inside the container
./scripts/test.sh --verbose
```

### Explore Examples

```bash
# Inside the container
cd examples/
python pytorch_integration.py
```

## Development Workflow

### Edit Code

Edit files on your host machine using your favorite IDE. Changes are automatically synced to the container.

### Build Changes

```bash
# Quick rebuild
docker compose exec tensorfuse ./scripts/build.sh

# Clean rebuild
docker compose exec tensorfuse ./scripts/build.sh --clean
```

### Run Tests

```bash
# All tests
docker compose exec tensorfuse ./scripts/test.sh

# C++ tests only
docker compose exec tensorfuse ./scripts/test-cpp.sh

# Python tests only
docker compose exec tensorfuse ./scripts/test-python.sh
```

## Common Tasks

### Check GPU Status

```bash
docker compose exec tensorfuse nvidia-smi
```

### Monitor Build Progress

```bash
docker compose logs -f tensorfuse
```

### Access Container Shell

```bash
docker compose exec tensorfuse bash
```

### Stop Environment

```bash
docker compose down
```

## Troubleshooting

### GPU Not Found

```bash
# Check NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.6-base nvidia-smi

# If this fails, reinstall NVIDIA Container Toolkit
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Build Errors

```bash
# Clean build
docker compose exec tensorfuse ./scripts/clean.sh
docker compose exec tensorfuse ./scripts/build.sh --clean

# Check dependencies
docker compose exec tensorfuse nvcc --version
docker compose exec tensorfuse find /usr/local -name "*cutlass*" -type d
```

### Container Issues

```bash
# Restart container
docker compose restart tensorfuse

# Rebuild container
docker compose down
docker compose build --no-cache
docker compose up -d
```

## Performance Tips

- Use `--parallel` flag for faster builds
- Monitor GPU memory with `nvidia-smi`
- Use Docker volumes for persistent data
- Keep containers running for faster development

## Getting Help

- Check the [full documentation](../README.md)
- Visit the [Docker guide](../docker/README.md)
- Open an issue on GitHub if you encounter problems

## Summary

You should now have:
- ✅ TensorFuse built and tested
- ✅ Development environment running
- ✅ Python bindings working
- ✅ GPU access confirmed

Happy coding with TensorFuse! 🚀 