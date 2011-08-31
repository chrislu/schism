#Project Name
project(schism)

# include platform definitions
include(schism_platform)
include(schism)

# Include subprojects
add_subdirectory(../../scm_core/build/cmake     ${CMAKE_CURRENT_BINARY_DIR}/scm_core_out)
add_subdirectory(../../scm_gl_core/build/cmake  ${CMAKE_CURRENT_BINARY_DIR}/scm_gl_core_out)
add_subdirectory(../../scm_gl_util/build/cmake  ${CMAKE_CURRENT_BINARY_DIR}/scm_gl_util_out)
add_subdirectory(../../scm_data/build/cmake     ${CMAKE_CURRENT_BINARY_DIR}/scm_data_out)
add_subdirectory(../../scm_input/build/cmake    ${CMAKE_CURRENT_BINARY_DIR}/scm_input_out)
