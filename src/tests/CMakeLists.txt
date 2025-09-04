# ==============================================================================
# TensorFuse Tests
# ==============================================================================

# Test executable for kernels
add_executable(test_kernels test_kernels.cu)
target_link_libraries(test_kernels tensorfuse)

# Add C API debug test
add_executable(test_c_api_debug test_c_api_debug.cpp)
target_link_libraries(test_c_api_debug ${CUDA_LIBRARIES} tensorfuse)
target_compile_features(test_c_api_debug PRIVATE cxx_std_17)
set_target_properties(test_c_api_debug PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add wrapper debug test
add_executable(test_wrapper_debug test_wrapper_debug.cpp)
target_link_libraries(test_wrapper_debug ${CUDA_LIBRARIES} tensorfuse)
target_compile_features(test_wrapper_debug PRIVATE cxx_std_17)
set_target_properties(test_wrapper_debug PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add existing tests
add_executable(test_simple_debug test_simple_debug.cu)
target_link_libraries(test_simple_debug ${CUDA_LIBRARIES} tensorfuse)
target_compile_features(test_simple_debug PRIVATE cxx_std_17)
set_target_properties(test_simple_debug PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Test executable for working kernel
add_executable(test_working_kernel test_working_kernel.cu)
target_link_libraries(test_working_kernel tensorfuse)

# Test executable for simple kernels
add_executable(test_kernels_simple test_kernels_simple.cpp)
target_link_libraries(test_kernels_simple tensorfuse)

# Test executable for softmax dropout
add_executable(test_softmax_dropout test_softmax_dropout.cu)
target_link_libraries(test_softmax_dropout tensorfuse)

# Test executable for GEMM edge cases
add_executable(test_gemm_edge_cases test_gemm_edge_cases.cu)
target_link_libraries(test_gemm_edge_cases tensorfuse)

# Add Python tests subdirectory
add_subdirectory(python)

# Set CUDA properties for all test executables
foreach(test_target test_kernels test_simple_debug test_working_kernel test_kernels_simple test_softmax_dropout test_gemm_edge_cases)
    set_target_properties(${test_target} PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
    )
endforeach()

# Enable testing
enable_testing()

# Add test cases
add_test(NAME test_kernels COMMAND test_kernels)
add_test(NAME test_simple_debug COMMAND test_simple_debug)
add_test(NAME test_working_kernel COMMAND test_working_kernel)
add_test(NAME test_kernels_simple COMMAND test_kernels_simple)
add_test(NAME test_softmax_dropout COMMAND test_softmax_dropout)
add_test(NAME test_gemm_edge_cases COMMAND test_gemm_edge_cases) 