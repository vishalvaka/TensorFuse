# FindCUTLASS.cmake
# Find CUTLASS library and headers
#
# This module defines:
#   CUTLASS_FOUND - True if CUTLASS is found
#   CUTLASS_INCLUDE_DIRS - Include directories for CUTLASS
#   CUTLASS_LIBRARIES - Libraries needed to use CUTLASS
#   CUTLASS_VERSION - Version of CUTLASS found

find_path(CUTLASS_INCLUDE_DIR
    NAMES cutlass/cutlass.h
    PATHS
        ${CUTLASS_ROOT_DIR}
        ${CUTLASS_ROOT_DIR}/include
        $ENV{CUTLASS_ROOT_DIR}
        $ENV{CUTLASS_ROOT_DIR}/include
        /usr/local/include
        /usr/include
        /opt/cutlass/include
        /opt/nvidia/cutlass/include
    PATH_SUFFIXES
        cutlass
)

# Check if we can find the version
if(CUTLASS_INCLUDE_DIR)
    file(READ ${CUTLASS_INCLUDE_DIR}/cutlass/version.h CUTLASS_VERSION_CONTENT)
    string(REGEX MATCH "#define CUTLASS_MAJOR_VERSION ([0-9]+)" _ "${CUTLASS_VERSION_CONTENT}")
    set(CUTLASS_VERSION_MAJOR "${CMAKE_MATCH_1}")
    string(REGEX MATCH "#define CUTLASS_MINOR_VERSION ([0-9]+)" _ "${CUTLASS_VERSION_CONTENT}")
    set(CUTLASS_VERSION_MINOR "${CMAKE_MATCH_1}")
    string(REGEX MATCH "#define CUTLASS_PATCH_VERSION ([0-9]+)" _ "${CUTLASS_VERSION_CONTENT}")
    set(CUTLASS_VERSION_PATCH "${CMAKE_MATCH_1}")
    
    if(CUTLASS_VERSION_MAJOR AND CUTLASS_VERSION_MINOR AND CUTLASS_VERSION_PATCH)
        set(CUTLASS_VERSION "${CUTLASS_VERSION_MAJOR}.${CUTLASS_VERSION_MINOR}.${CUTLASS_VERSION_PATCH}")
    endif()
endif()

# CUTLASS is header-only, so we don't need to find libraries
set(CUTLASS_LIBRARIES "")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUTLASS
    FOUND_VAR CUTLASS_FOUND
    REQUIRED_VARS CUTLASS_INCLUDE_DIR
    VERSION_VAR CUTLASS_VERSION
)

if(CUTLASS_FOUND)
    set(CUTLASS_INCLUDE_DIRS ${CUTLASS_INCLUDE_DIR})
    
    # Create interface target
    if(NOT TARGET CUTLASS::CUTLASS)
        add_library(CUTLASS::CUTLASS INTERFACE IMPORTED)
        set_target_properties(CUTLASS::CUTLASS PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${CUTLASS_INCLUDE_DIR}"
        )
        
        # Add required CUDA flags for CUTLASS
        set_target_properties(CUTLASS::CUTLASS PROPERTIES
            INTERFACE_COMPILE_OPTIONS 
                "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda;--use_fast_math>"
        )
    endif()
endif()

mark_as_advanced(CUTLASS_INCLUDE_DIR) 