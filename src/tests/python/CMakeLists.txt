# ==============================================================================
# TensorFuse Python Tests
# ==============================================================================

# Python tests are run via pytest, not through CMake
# This is a placeholder CMakeLists.txt file

# Add custom target for running Python tests
add_custom_target(python_tests
    COMMAND python3 -m pytest ${CMAKE_CURRENT_SOURCE_DIR} -v
    DEPENDS tensorfuse_python
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running Python tests"
)

# Add test for each Python test file
file(GLOB PYTHON_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/test_*.py")

foreach(test_file ${PYTHON_TEST_FILES})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_test(
        NAME ${test_name}
        COMMAND python3 -m pytest ${test_file} -v
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endforeach()

message(STATUS "Python tests configuration complete") 