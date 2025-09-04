# ==============================================================================
# TensorFuse Benchmarks
# ==============================================================================

# Create results directory
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/results)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Common benchmark settings
set(BENCHMARK_COMMON_SOURCES "tensorfuse_wrappers.cpp")
set(BENCHMARK_COMMON_LIBRARIES 
    tensorfuse
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    CUDA::curand
)

# Add compiler flags for benchmarks
set(BENCHMARK_COMPILE_FLAGS 
    --expt-relaxed-constexpr
    --use_fast_math
    -O3
)

# ==============================================================================
# BERT Baseline Benchmark
# ==============================================================================

add_executable(bert_baseline
    bert_baseline.cu
    ${BENCHMARK_COMMON_SOURCES}
)

target_include_directories(bert_baseline PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(bert_baseline
    ${BENCHMARK_COMMON_LIBRARIES}
)

set_target_properties(bert_baseline PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

target_compile_options(bert_baseline PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${BENCHMARK_COMPILE_FLAGS}>
)

# ==============================================================================
# Roofline Analysis Benchmark
# ==============================================================================

add_executable(roofline_analysis
    roofline_analysis.cu
    ${BENCHMARK_COMMON_SOURCES}
)

target_include_directories(roofline_analysis PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(roofline_analysis
    ${BENCHMARK_COMMON_LIBRARIES}
)

set_target_properties(roofline_analysis PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

target_compile_options(roofline_analysis PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${BENCHMARK_COMPILE_FLAGS}>
)

# ==============================================================================
# Make nsight profiling script executable
# ==============================================================================

# Copy the nsight profiling script to build directory
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/nsight_trace.sh
    ${CMAKE_CURRENT_BINARY_DIR}/nsight_trace.sh
    COPYONLY
)

# Make it executable
file(CHMOD ${CMAKE_CURRENT_BINARY_DIR}/nsight_trace.sh
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE
)

# ==============================================================================
# Benchmark targets
# ==============================================================================

# Add custom targets for running benchmarks
add_custom_target(run_bert_baseline
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bert_baseline
    DEPENDS bert_baseline
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running BERT baseline benchmark"
)

add_custom_target(run_roofline_analysis
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/roofline_analysis
    DEPENDS roofline_analysis
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running roofline analysis"
)

add_custom_target(run_nsight_profiling
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/nsight_trace.sh
    DEPENDS bert_baseline roofline_analysis
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running comprehensive Nsight Systems profiling"
)

# Combined benchmark target
add_custom_target(run_all_benchmarks
    DEPENDS run_bert_baseline run_roofline_analysis
    COMMENT "Running all benchmarks"
)

# ==============================================================================
# Install benchmarks (optional)
# ==============================================================================

install(TARGETS bert_baseline roofline_analysis
    RUNTIME DESTINATION bin
    COMPONENT benchmarks
)

install(FILES nsight_trace.sh
    DESTINATION bin
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE
    COMPONENT benchmarks
)

message(STATUS "Benchmarks configuration complete")
message(STATUS "  - BERT baseline benchmark: bert_baseline")
message(STATUS "  - Roofline analysis: roofline_analysis")
message(STATUS "  - Nsight profiling: nsight_trace.sh")
message(STATUS "  - Build targets: run_bert_baseline, run_roofline_analysis, run_nsight_profiling, run_all_benchmarks") 