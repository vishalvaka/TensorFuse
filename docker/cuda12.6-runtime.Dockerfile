# ==============================================================================
# TensorFuse CUDA 12.6 Runtime Environment
# ==============================================================================

# Base image with CUDA 12.6 runtime
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    pkg-config \
    # Python development
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    # Development tools
    gdb \
    valgrind \
    clang-tidy \
    clang-format \
    # Documentation tools
    doxygen \
    graphviz \
    # Utilities
    htop \
    tmux \
    vim \
    nano \
    tree \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install cuDNN development libraries
RUN apt-get update && apt-get install -y \
    libcudnn8-dev \
    libcudnn8 \
    && rm -rf /var/lib/apt/lists/*

# Install Nsight profiling tools (specific versions)
RUN apt-get update && apt-get install -y \
    nsight-systems-2024.6.2 \
    nsight-compute-2024.3.2 \
    && rm -rf /var/lib/apt/lists/* || true

# Install modern CMake
RUN CMAKE_VERSION="3.27.7" && \
    wget -O cmake.sh https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    chmod +x cmake.sh && \
    ./cmake.sh --skip-license --prefix=/usr/local && \
    rm cmake.sh

# Install CUTLASS
RUN CUTLASS_VERSION="3.5.1" && \
    git clone --branch v${CUTLASS_VERSION} --depth 1 https://github.com/NVIDIA/cutlass.git /tmp/cutlass && \
    cd /tmp/cutlass && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCUTLASS_ENABLE_HEADERS_ONLY=ON \
        -DCUTLASS_ENABLE_TOOLS=OFF \
        -DCUTLASS_ENABLE_EXAMPLES=OFF \
        -DCUTLASS_ENABLE_TESTS=OFF && \
    make install -j$(nproc) && \
    cd / && rm -rf /tmp/cutlass

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Install pybind11
RUN python3 -m pip install --no-cache-dir pybind11[global]

# Install Transformer Engine (for FP8 support) - optional for now
# RUN python3 -m pip install --no-cache-dir transformer-engine[pytorch]

# Install Google Test
RUN git clone --branch v1.14.0 --depth 1 https://github.com/google/googletest.git /tmp/googletest && \
    cd /tmp/googletest && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_GMOCK=ON \
        -DGTEST_HAS_PTHREAD=ON && \
    make install -j$(nproc) && \
    cd / && rm -rf /tmp/googletest

# Install Conan package manager
RUN python3 -m pip install --no-cache-dir conan

# Set up workspace
WORKDIR /workspace

# Copy project files
COPY . .

# Create build directory
RUN mkdir -p build

# Set default environment variables for development
ENV TENSORFUSE_CUDA_ARCHITECTURES="89;90"
ENV TENSORFUSE_BUILD_TYPE="Release"
ENV TENSORFUSE_ENABLE_PROFILING=ON
ENV TENSORFUSE_ENABLE_FP8=ON

# Expose common ports for development
EXPOSE 8888 8889 8890

# Add development aliases
RUN echo 'alias ll="ls -la"' >> ~/.bashrc && \
    echo 'alias la="ls -A"' >> ~/.bashrc && \
    echo 'alias l="ls -CF"' >> ~/.bashrc && \
    echo 'alias build="cd /workspace && cmake --build build"' >> ~/.bashrc && \
    echo 'alias test="cd /workspace && ctest --test-dir build"' >> ~/.bashrc && \
    echo 'alias clean="cd /workspace && rm -rf build && mkdir build"' >> ~/.bashrc

# Entry point script
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set up git configuration (for development)
RUN git config --global user.name "TensorFuse Developer" && \
    git config --global user.email "developer@tensorfuse.ai" && \
    git config --global init.defaultBranch main

# Configure CUDA environment
RUN echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc && \
    echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc

# Install development utilities
RUN python3 -m pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    matplotlib \
    seaborn \
    plotly \
    pandas \
    numpy \
    scipy \
    scikit-learn

# Create cache directories
RUN mkdir -p /workspace/cache/autotuner && \
    mkdir -p /workspace/cache/kernels && \
    mkdir -p /workspace/benchmarks/results

# Set permissions
RUN chmod -R 755 /workspace

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"] 