
# Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
# Distributed under the Modified BSD License, see license.txt.

# set project name
project(schism)

cmake_minimum_required(VERSION 2.8)

# tell cmake where to look for our scripts
set(CMAKE_MODULE_PATH ${schism_SOURCE_DIR}/custom_scripts ${CMAKE_MODULE_PATH})

include(schism_platform)
include(schism)
include(schism_compiler)

# set some directory constants
set(GLOBAL_EXT_DIR ${schism_SOURCE_DIR}/../../externals)

# include subprojects
add_subdirectory(../../scm_core/build/cmake     ${CMAKE_CURRENT_BINARY_DIR}/scm_core_out)
add_subdirectory(../../scm_cl_core/build/cmake  ${CMAKE_CURRENT_BINARY_DIR}/scm_cl_core_out)
add_subdirectory(../../scm_gl_core/build/cmake  ${CMAKE_CURRENT_BINARY_DIR}/scm_gl_core_out)
add_subdirectory(../../scm_gl_util/build/cmake  ${CMAKE_CURRENT_BINARY_DIR}/scm_gl_util_out)
add_subdirectory(../../scm_data/build/cmake     ${CMAKE_CURRENT_BINARY_DIR}/scm_data_out)
add_subdirectory(../../scm_input/build/cmake    ${CMAKE_CURRENT_BINARY_DIR}/scm_input_out)

# add subdirectories to projects
add_subdirectory(${schism_SOURCE_DIR}/../../projects/build/cmake       ${CMAKE_BINARY_DIR}/projects_cmake_out)
