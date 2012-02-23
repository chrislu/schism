
# Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
# Distributed under the Modified BSD License, see license.txt.

project(scm_data)

include(schism)
include(schism_macros)
include(schism_boost)

set(SRC_DIR                         "${scm_date_SOURCE_DIR}../../src/scm")
set(SCM_PROJECT_SOURCE_DIR          ${SRC_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${SCHISM_LIBRARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${SCHISM_LIBRARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${SCHISM_LIBRARY_DIR})

# source files
scm_project_files(SOURCE_FILES      ${SRC_DIR} *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR} *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/data *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/data *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/data/analysis *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/data/analysis *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/data/analysis/histogram *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/data/analysis/histogram *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/data/analysis/transfer_function *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/data/analysis/transfer_function *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/data/analysis/processing *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/data/analysis/processing *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/data/geometry *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/data/geometry *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/data/geometry/scm_geom *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/data/geometry/scm_geom *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/data/geometry/wavefront_obj *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/data/geometry/wavefront_obj *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/data/volume *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/data/volume *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/data/volume/voxel_geo_vol *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/data/volume/voxel_geo_vol *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/data/volume/scm_vol *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/data/volume/scm_vol *.h *.inl)

# include header and inline files in source files for visual studio projects
if (WIN32)
    if (MSVC)
        set (SOURCE_FILES ${SOURCE_FILES} ${HEADER_FILES})
    endif (MSVC)
endif (WIN32)

# set include and lib directories
scm_project_include_directories(ALL   ${SCM_ROOT_DIR}/scm_core/src
                                      ${SCM_ROOT_DIR}/scm_data/src
                                      ${SCM_BOOST_INC_DIR})
scm_project_include_directories(WIN32 ${GLOBAL_EXT_DIR}/inc)
#scm_project_include_directories(UNIX  )

# set library directories
scm_project_link_directories(ALL   ${SCM_LIB_DIR}/${SCHISM_PLATFORM}
                                   ${SCM_BOOST_LIB_DIR})
scm_project_link_directories(WIN32 ${GLOBAL_EXT_DIR}/lib)
#scm_project_link_directories(UNIX  )

if (WIN32)
    # add preprocessor defines
    add_definitions(-DSCM_BUILD_LIBRARY_data=\(\))
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
)
scm_link_libraries(WIN32
    optimized libboost_filesystem-${SCM_BOOST_MT_REL}   debug libboost_filesystem-${SCM_BOOST_MT_DBG}
    optimized libboost_system-${SCM_BOOST_MT_REL}       debug libboost_system-${SCM_BOOST_MT_DBG}
)
scm_link_libraries(UNIX
    boost_filesystem${SCM_BOOST_MT_REL}
    boost_system${SCM_BOOST_MT_REL}
)

add_dependencies(${PROJECT_NAME}
    scm_core
)

