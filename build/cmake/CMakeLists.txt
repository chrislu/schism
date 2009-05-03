#Project Name
project(schism)

# include platform definitions
include(schism_platform)
include(schism)

#-----------------------------------------------------------------------------
# Setup schism_VR version strings
set(SCHISM_MAJOR_VERSION 0)
set(SCHISM_MINOR_VERSION 0)
set(SCHISM_BUILD_VERSION 1)
set(SCHISM_VERSION "${schism_MAJOR_VERSION}.${schism_MINOR_VERSION}.${schism_BUILD_VERSION}")

#-----------------------------------------------------------------------------
# setup output directories.
if(NOT LIBRARY_OUTPUT_PATH)
  set(LIBRARY_OUTPUT_PATH ${schism_SOURCE_DIR}/../../lib/${SCHISM_PLATFORM} CACHE INTERNAL "Single output directory for building all libraries.")
endif(NOT LIBRARY_OUTPUT_PATH)

if(NOT EXECUTABLE_OUTPUT_PATH)
  set(EXECUTABLE_OUTPUT_PATH ${schism_SOURCE_DIR}/../../bin/${SCHISM_PLATFORM} CACHE INTERNAL "Single output directory for building all executables.")
endif(NOT EXECUTABLE_OUTPUT_PATH)

set(SCHISM_LIBRARY_DIR    ${LIBRARY_OUTPUT_PATH})
set(SCHISM_EXECUTABLE_DIR ${EXECUTABLE_OUTPUT_PATH})

# Include subprojects
add_subdirectory(../../scm_core/build/cmake     ${CMAKE_CURRENT_BINARY_DIR}/scm_core_cmake_out)
add_subdirectory(../../scm_gl/build/cmake       ${CMAKE_CURRENT_BINARY_DIR}/scm_gl_cmake_out)
add_subdirectory(../../scm_data/build/cmake     ${CMAKE_CURRENT_BINARY_DIR}/scm_data_cmake_out)
add_subdirectory(../../scm_input/build/cmake    ${CMAKE_CURRENT_BINARY_DIR}/scm_input_cmake_out)
