cmake_minimum_required(VERSION 3.22)
project(TensorFuse 
    VERSION 1.0.0
    LANGUAGES C CXX CUDA
    DESCRIPTION "Tensor-Core-Optimized Transformer Inference Library"
)

# ==============================================================================
# Build Configuration
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA Configuration
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ==============================================================================
# Options
# ==============================================================================
option(TENSORFUSE_BUILD_TESTS "Build tests" ON)
option(TENSORFUSE_BUILD_PYTHON "Build Python bindings" ON)
option(TENSORFUSE_BUILD_BENCHMARKS "Build benchmarks" ON)
option(TENSORFUSE_BUILD_DOCS "Build documentation" OFF)
option(TENSORFUSE_ENABLE_PROFILING "Enable profiling support" ON)
option(TENSORFUSE_ENABLE_FP8 "Enable FP8 support (requires Hopper+)" ON)

# ==============================================================================
# CUDA Architecture Detection
# ==============================================================================
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Default to Ada and Hopper architectures
    set(CMAKE_CUDA_ARCHITECTURES "89;90")
endif()

# ==============================================================================
# Dependencies
# ==============================================================================
find_package(CUDAToolkit 12.4 REQUIRED)

# Find CUTLASS
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(CUTLASS 3.5 REQUIRED)

# Transformer Engine (optional, for FP8 support)
if(TENSORFUSE_ENABLE_FP8)
    find_package(TransformerEngine 1.9 QUIET)
    if(NOT TransformerEngine_FOUND)
        message(WARNING "Transformer Engine not found. FP8 support will be disabled.")
        set(TENSORFUSE_ENABLE_FP8 OFF)
    endif()
endif()

# ==============================================================================
# Compiler Flags
# ==============================================================================
# C++ Flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Werror")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

# CUDA Flags
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math -lineinfo")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG --use_fast_math -lineinfo")

# Add common CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")

# ==============================================================================
# Include Directories
# ==============================================================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime
    ${CUDAToolkit_INCLUDE_DIRS}
)

# ==============================================================================
# Subdirectories
# ==============================================================================
add_subdirectory(src)

if(TENSORFUSE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(src/tests)
endif()

if(TENSORFUSE_BUILD_PYTHON)
    add_subdirectory(src/python)
endif()

if(TENSORFUSE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(TENSORFUSE_BUILD_DOCS)
    add_subdirectory(docs)
endif()

# ==============================================================================
# Installation
# ==============================================================================
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Configure package config file
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TensorFuseConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/TensorFuseConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorFuse
)

# Install config files
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TensorFuseConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorFuse
)

# ==============================================================================
# Package Configuration
# ==============================================================================
set(CPACK_PACKAGE_NAME "TensorFuse")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PROJECT_DESCRIPTION})
set(CPACK_PACKAGE_VENDOR "TensorFuse Development Team")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

include(CPack) 