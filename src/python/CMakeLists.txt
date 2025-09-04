# ==============================================================================
# TensorFuse Python Bindings
# ==============================================================================

# Find required dependencies
find_package(pybind11 REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# ==============================================================================
# Python Module Source Files
# ==============================================================================

set(PYTHON_BINDING_SOURCES
    tensorfuse_python.cpp
    core_bindings.cpp
    memory_bindings.cpp
    tensor_bindings.cpp
    profiler_bindings.cpp
)

# ==============================================================================
# Create Python Module
# ==============================================================================

pybind11_add_module(tensorfuse_python 
    ${PYTHON_BINDING_SOURCES}
)

# Set module properties
set_target_properties(tensorfuse_python PROPERTIES
    OUTPUT_NAME "_tensorfuse"
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
)

# ==============================================================================
# Link Dependencies
# ==============================================================================

target_link_libraries(tensorfuse_python
    PRIVATE
        tensorfuse
        pybind11::module
        pybind11::lto
        pybind11::windows_extras
)

# Configure LTO to use parallel compilation
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_options(tensorfuse_python PRIVATE -flto=auto)
endif()

# ==============================================================================
# Include Directories
# ==============================================================================

target_include_directories(tensorfuse_python
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../tensorfuse
        ${CMAKE_CURRENT_SOURCE_DIR}/../runtime
        ${CMAKE_CURRENT_SOURCE_DIR}/../utils
        ${Python_INCLUDE_DIRS}
)

# ==============================================================================
# Compile Definitions
# ==============================================================================

target_compile_definitions(tensorfuse_python
    PRIVATE
        VERSION_INFO="${PROJECT_VERSION}"
        PYBIND11_DETAILED_ERROR_MESSAGES
)

# Add profiling flag to match main library
if(TENSORFUSE_ENABLE_PROFILING)
    target_compile_definitions(tensorfuse_python
        PRIVATE
            TENSORFUSE_ENABLE_PROFILING=1
    )
else()
    target_compile_definitions(tensorfuse_python
        PRIVATE
            TENSORFUSE_ENABLE_PROFILING=0
    )
endif()

# ==============================================================================
# CUDA Support
# ==============================================================================

if(CMAKE_CUDA_COMPILER)
    set_target_properties(tensorfuse_python PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# ==============================================================================
# Installation
# ==============================================================================

# Install the module
install(TARGETS tensorfuse_python
    COMPONENT python
    LIBRARY DESTINATION "."
)

# Install Python package files
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/tensorfuse/__init__.py
    ${CMAKE_CURRENT_SOURCE_DIR}/tensorfuse/core.py
    ${CMAKE_CURRENT_SOURCE_DIR}/tensorfuse/memory.py
    ${CMAKE_CURRENT_SOURCE_DIR}/tensorfuse/profiler.py
    ${CMAKE_CURRENT_SOURCE_DIR}/tensorfuse/utils.py
    DESTINATION tensorfuse
    COMPONENT python
)

# Install setup.py for development
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/setup.py
    DESTINATION "."
    COMPONENT python
) 