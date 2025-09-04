# ==============================================================================
# TensorFuse Library Source
# ==============================================================================

# ==============================================================================
# Source Files
# ==============================================================================

# Core library headers
set(TENSORFUSE_HEADERS
    tensorfuse.h
    tensorfuse/types.h
    tensorfuse/config.h
    tensorfuse/profiler.h
    kernels/fused_gemm_bias_gelu.h
    kernels/fused_softmax_dropout.h
    kernels/kernel_traits.h
    runtime/memory_manager.h
    runtime/runtime_context.h
    utils/error_handling.h
    utils/cuda_utils.h
)

# Core library sources
set(TENSORFUSE_SOURCES
    runtime/memory_manager.cpp
    runtime/runtime_context.cpp
    utils/error_handling.cpp
    tensorfuse_c_api.cpp
)

# CUDA kernel sources
set(TENSORFUSE_CUDA_SOURCES
    kernels/fused_gemm_bias_gelu.cu
    kernels/fused_gemm_bias_gelu_optimized.cu
    kernels/fused_gemm_bias_gelu_int8.cu
    kernels/fused_softmax_dropout.cu
)

# ==============================================================================
# Library Target
# ==============================================================================

# Create the main library
add_library(tensorfuse SHARED
    ${TENSORFUSE_SOURCES}
    ${TENSORFUSE_CUDA_SOURCES}
    ${TENSORFUSE_HEADERS}
)

# Add alias for consistent naming
add_library(TensorFuse::tensorfuse ALIAS tensorfuse)

# ==============================================================================
# Target Properties
# ==============================================================================

# Set library properties
set_target_properties(tensorfuse PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# ==============================================================================
# Dependencies
# ==============================================================================

# Link CUDA libraries
target_link_libraries(tensorfuse
    PUBLIC
        CUDA::toolkit
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
        CUDA::cusparse
        CUDA::cufft
        cudnn
)

# Link CUTLASS
target_link_libraries(tensorfuse
    PUBLIC
        CUTLASS::CUTLASS
)

# Optional: Link Transformer Engine for FP8 support
if(TENSORFUSE_ENABLE_FP8 AND TransformerEngine_FOUND)
    target_link_libraries(tensorfuse
        PUBLIC
            TransformerEngine::transformer_engine
    )
    target_compile_definitions(tensorfuse
        PUBLIC
            TENSORFUSE_ENABLE_FP8=1
    )
endif()

# ==============================================================================
# Compile Features and Definitions
# ==============================================================================

# Set compile features
target_compile_features(tensorfuse
    PUBLIC
        cxx_std_17
        cuda_std_17
)

# Add compile definitions
target_compile_definitions(tensorfuse
    PUBLIC
        TENSORFUSE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        TENSORFUSE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        TENSORFUSE_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# Add profiling support if enabled
if(TENSORFUSE_ENABLE_PROFILING)
    target_compile_definitions(tensorfuse
        PUBLIC
            TENSORFUSE_ENABLE_PROFILING=1
    )
else()
    target_compile_definitions(tensorfuse
        PUBLIC
            TENSORFUSE_ENABLE_PROFILING=0
    )
endif()

# ==============================================================================
# Include Directories
# ==============================================================================

target_include_directories(tensorfuse
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# ==============================================================================
# CUDA Architecture Settings
# ==============================================================================

# Set CUDA architectures for the library
set_property(TARGET tensorfuse 
    PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

# ==============================================================================
# Installation
# ==============================================================================

# Install the library
install(TARGETS tensorfuse
    EXPORT TensorFuseTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(FILES ${TENSORFUSE_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tensorfuse
    COMPONENT Development
)

# Install additional headers preserving directory structure
install(DIRECTORY .
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tensorfuse
    COMPONENT Development
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.cuh"
)

# Export targets
install(EXPORT TensorFuseTargets
    FILE TensorFuseTargets.cmake
    NAMESPACE TensorFuse::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorFuse
    COMPONENT Development
)

# ==============================================================================
# Debug Information
# ==============================================================================

# Print build configuration
message(STATUS "TensorFuse Library Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  FP8 Support: ${TENSORFUSE_ENABLE_FP8}")
message(STATUS "  Profiling Support: ${TENSORFUSE_ENABLE_PROFILING}")
message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}") 