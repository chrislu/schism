
# Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
# Distributed under the Modified BSD License, see license.txt.

project(scm_cl_core)

include(schism_macros)
include(schism_boost)
include(schism_cuda)

set(SRC_DIR                         "${scm_cl_core_SOURCE_DIR}/../../src/scm")
set(SCM_PROJECT_SOURCE_DIR          ${SRC_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${SCHISM_LIBRARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${SCHISM_LIBRARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${SCHISM_LIBRARY_DIR})

# source files
scm_project_files(SOURCE_FILES      ${SRC_DIR} *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR} *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/cl_core *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/cl_core *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/cl_core/opencl *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/cl_core/opencl *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/cl_core/cuda *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/cl_core/cuda *.h *.inl)

# include header and inline files in source files for visual studio projects
if (WIN32)
    if (MSVC)
        set (SOURCE_FILES ${SOURCE_FILES} ${HEADER_FILES})
    endif (MSVC)
endif (WIN32)


# set include and lib directories
scm_project_include_directories(ALL   ${SCM_ROOT_DIR}/scm_core/src
                                      ${SCM_ROOT_DIR}/scm_cl_core/src
                                      ${SCM_BOOST_INC_DIR})
scm_project_include_directories(WIN32 ${GLOBAL_EXT_DIR}/inc
	                                  ${GLOBAL_EXT_DIR}/inc/cuda)
scm_project_include_directories(UNIX /opt/OpenCL/include
	                             /opt/cuda/current/cuda/include)

# set library directories
scm_project_link_directories(ALL   ${SCM_LIB_DIR}/${SCHISM_PLATFORM}
                                   ${SCM_BOOST_LIB_DIR})
scm_project_link_directories(WIN32 ${GLOBAL_EXT_DIR}/lib)
scm_project_link_directories(UNIX  /opt/cuda/current/cuda/lib64
                                   /usr/lib/nvidia-current)

if (WIN32)
    # add preprocessor defines
    add_definitions(-DSCM_BUILD_LIBRARY_cl_core=\(\))
endif (WIN32)

# add/create library
if (SCHISM_BUILD_STATIC)
    add_definitions(-DSCM_STATIC_BUILD)
	add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES})
else (SCHISM_BUILD_STATIC)
	add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
endif (SCHISM_BUILD_STATIC)

# link libraries
scm_link_libraries(ALL
	scm_core

	general cuda
	general cudart
)
scm_link_libraries(WIN32
    general OpenCL
    general opengl32
)
scm_link_libraries(UNIX
    general GL
    general OpenCL

    general cuda
    general cudart
)

add_dependencies(${PROJECT_NAME}
    scm_core
)

