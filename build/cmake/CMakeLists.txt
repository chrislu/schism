
# Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
# Distributed under the Modified BSD License, see license.txt.

# set project name
project(schism)

cmake_minimum_required(VERSION 3.0.2)

# tell cmake where to look for our scripts
set(CMAKE_MODULE_PATH ${schism_SOURCE_DIR}/custom_scripts ${CMAKE_MODULE_PATH})

include(schism_platform)
include(schism)
include(schism_compiler)

option(SCM_ENABLE_CUDA_CL_SUPPORT         "Enable and OpenCL/CUDA functionality and OpenGL interoperability."                OFF)

# set some directory constants
set(GLOBAL_EXT_DIR ${schism_SOURCE_DIR}/../../externals)

# set debug library extension
set(SCHISM_DEBUG_LIBRARY_EXTENSION -gd)

# include subprojects
add_subdirectory(../../scm_core/build/cmake     ${CMAKE_CURRENT_BINARY_DIR}/scm_core_out)
add_subdirectory(../../scm_gl_core/build/cmake  ${CMAKE_CURRENT_BINARY_DIR}/scm_gl_core_out)
add_subdirectory(../../scm_gl_util/build/cmake  ${CMAKE_CURRENT_BINARY_DIR}/scm_gl_util_out)
add_subdirectory(../../scm_input/build/cmake    ${CMAKE_CURRENT_BINARY_DIR}/scm_input_out)

IF (${SCM_ENABLE_CUDA_CL_SUPPORT})
  include(schism_cuda)
  add_subdirectory(../../scm_cl_core/build/cmake  ${CMAKE_CURRENT_BINARY_DIR}/scm_cl_core_out)
ELSE (${SCM_ENABLE_CUDA_CL_SUPPORT})
  unset(SCHISM_CUDA_BUILD_COMPUTE20 CACHE)
  unset(SCHISM_CUDA_BUILD_COMPUTE30 CACHE)
  unset(SCHISM_CUDA_BUILD_COMPUTE35 CACHE)
  unset(SCHISM_CUDA_BUILD_COMPUTE50 CACHE)
  unset(SCHISM_CUDA_BUILD_COMPUTE52 CACHE)
  unset(SCHISM_CUDA_BUILD_VERBOSE CACHE)
  unset(SCHISM_CUDA_BUILD_USE_FAST_MATH CACHE)
  unset(SCHISM_CUDA_BUILD_KEEP_INTERMEDIATE_FILES CACHE)
  unset(SCM_CUDA_SHARED_LIB_NAME CACHE)
  unset(SCM_CUDA_NVCC_OPTIONS CACHE)
  unset(SCM_CUDA_NVCC_COMMAND CACHE)
  unset(SCM_CUDA_NVCC_PATH CACHE)
ENDIF (${SCM_ENABLE_CUDA_CL_SUPPORT})

# add subdirectories to projects
add_subdirectory(${schism_SOURCE_DIR}/../../projects/build/cmake       ${CMAKE_BINARY_DIR}/projects_cmake_out)
