
# Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
# Distributed under the Modified BSD License, see license.txt.

project(scm_gl_util)

include(schism_macros)
include(schism_boost)

set(SRC_DIR                         "${scm_gl_util_SOURCE_DIR}/../../src/scm")
set(SCM_PROJECT_SOURCE_DIR          ${SRC_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${SCHISM_LIBRARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${SCHISM_LIBRARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${SCHISM_LIBRARY_DIR})

# source files
scm_project_files(SOURCE_FILES      ${SRC_DIR} *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR} *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_util *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_util *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_util/font *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_util/font *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_util/font/detail *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_util/font/detail *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_util/data/analysis *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_util/data/analysis *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_util/data/analysis/transfer_function *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_util/data/analysis/transfer_function *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_util/data/imaging *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_util/data/imaging *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_util/data/volume *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_util/data/volume *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_util/data/volume/voxel_geo *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_util/data/volume/voxel_geo *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_util/data/volume/segy *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_util/data/volume/segy *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_util/manipulators *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_util/manipulators *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_util/primitives *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_util/primitives *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_util/primitives/util *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_util/primitives/util *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_util/utilities *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_util/utilities *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_util/viewer *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_util/viewer *.h *.inl)


# include header and inline files in source files for visual studio projects
if (WIN32)
    if (MSVC)
        set (SOURCE_FILES ${SOURCE_FILES} ${HEADER_FILES})
    endif (MSVC)
endif (WIN32)


# set include and lib directories
scm_project_include_directories(ALL   ${SCM_ROOT_DIR}/scm_core/src
                                      ${SCM_ROOT_DIR}/scm_cl_core/src
                                      ${SCM_ROOT_DIR}/scm_gl_core/src
                                      ${SCM_ROOT_DIR}/scm_gl_util/src
                                      ${SCM_ROOT_DIR}/scm_input/src
                                      ${SCM_BOOST_INC_DIR})

scm_project_include_directories(WIN32 ${GLOBAL_EXT_DIR}/inc
                                      ${GLOBAL_EXT_DIR}/inc/freeimage
                                      ${GLOBAL_EXT_DIR}/inc/freetype
									  ${GLOBAL_EXT_DIR}/inc/cuda)

scm_project_include_directories(UNIX  /usr/include/freetype2
									  /opt/OpenCL/include
									  /opt/cuda/current/cuda/include)

# set library directories
scm_project_link_directories(ALL   ${SCM_LIB_DIR}/${SCHISM_PLATFORM}
                                   ${SCM_BOOST_LIB_DIR})
scm_project_link_directories(WIN32 ${GLOBAL_EXT_DIR}/lib)
scm_project_link_directories(UNIX  /opt/cuda/current/cuda/lib64)

if (WIN32)
    # add preprocessor defines
    add_definitions(-DSCM_BUILD_LIBRARY_gl_util=\(\))
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
    scm_gl_core
    scm_input

)
scm_link_libraries(WIN32
    scm_cl_core
    FreeImagePlus
    freetype2
	general cuda
	general cudart
    general OpenCL
)
scm_link_libraries(UNIX
    freeimageplus
    freetype
)

add_dependencies(${PROJECT_NAME}
    scm_core
    scm_cl_core
    scm_gl_core
    scm_input
)
