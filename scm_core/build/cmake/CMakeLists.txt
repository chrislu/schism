
# Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
# Distributed under the Modified BSD License, see license.txt.

project(scm_core)

include(schism_macros)
include(schism_boost)

set(SRC_DIR                         "${scm_core_SOURCE_DIR}/../../src/scm")
set(SCM_PROJECT_SOURCE_DIR          ${SRC_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${SCHISM_LIBRARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${SCHISM_LIBRARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${SCHISM_LIBRARY_DIR})

scm_project_files(SOURCE_FILES      ${SRC_DIR} *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR} *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/core *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/core *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/core/io *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/core/io *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/core/io/detail *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/core/io/detail *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/core/log *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/core/log *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/core/math *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/core/math *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/core/math/detail *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/core/math/detail *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/core/module *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/core/module *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/core/platform *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/core/platform *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/core/platform/graphics *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/core/platform/graphics *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/core/platform/graphics/detail *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/core/platform/graphics/detail *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/core/resource *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/core/resource *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/core/resource/detail *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/core/resource/detail *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/core/script *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/core/script *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/core/script/detail *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/core/script/detail *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/core/time *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/core/time *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/core/time/detail *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/core/time/detail *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/core/utilities *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/core/utilities *.h *.inl)

# include header and inline files in source files for visual studio projects
if (WIN32)
    if (MSVC)
        set (SOURCE_FILES ${SOURCE_FILES} ${HEADER_FILES})
    endif (MSVC)
endif (WIN32)

# set include and lib directories
scm_project_include_directories(ALL   ${SCM_ROOT_DIR}/scm_core/src
                                      ${SCM_BOOST_INC_DIR})
scm_project_include_directories(WIN32 ${GLOBAL_EXT_DIR}/inc)

scm_project_link_directories(ALL   ${SCM_LIB_DIR}/${SCHISM_PLATFORM}
                                   ${SCM_BOOST_LIB_DIR})
scm_project_link_directories(WIN32 ${GLOBAL_EXT_DIR}/lib)
#scm_project_link_directories(UNIX )

if (WIN32)
    # add preprocessor defines
    add_definitions(-DSCM_BUILD_LIBRARY_core=\(\))
endif (WIN32)

# add/create library
if (SCHISM_BUILD_STATIC)
    add_definitions(-DSCM_STATIC_BUILD)
	add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES})
else (SCHISM_BUILD_STATIC)
	add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
endif (SCHISM_BUILD_STATIC)

# link libraries
scm_link_libraries(WIN32
    optimized libboost_chrono-${SCM_BOOST_MT_REL}           debug libboost_chrono-${SCM_BOOST_MT_DBG}
    optimized libboost_date_time-${SCM_BOOST_MT_REL}        debug libboost_date_time-${SCM_BOOST_MT_DBG}
    optimized libboost_filesystem-${SCM_BOOST_MT_REL}       debug libboost_filesystem-${SCM_BOOST_MT_DBG}
    optimized libboost_program_options-${SCM_BOOST_MT_REL}  debug libboost_program_options-${SCM_BOOST_MT_DBG}
    optimized libboost_system-${SCM_BOOST_MT_REL}           debug libboost_system-${SCM_BOOST_MT_DBG}
    optimized libboost_timer-${SCM_BOOST_MT_REL}            debug libboost_timer-${SCM_BOOST_MT_DBG}
)
scm_link_libraries(UNIX
    rt
    boost_chrono${SCM_BOOST_MT_REL}
    boost_date_time${SCM_BOOST_MT_REL}
    boost_filesystem${SCM_BOOST_MT_REL}
    boost_program_options${SCM_BOOST_MT_REL}
    boost_system${SCM_BOOST_MT_REL}
    boost_timer${SCM_BOOST_MT_REL}
)
