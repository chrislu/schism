
project(scm_core)

include(schism_macros)
include(schism_boost)

set(SRC_DIR                         "${scm_core_SOURCE_DIR}/../../src/scm")
set(SCM_PROJECT_SOURCE_DIR          ${SRC_DIR})

SET(LIBRARY_OUTPUT_PATH             ${SCHISM_LIBRARY_DIR})

scm_project_files(SOURCE_FILES      ${SRC_DIR} *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR} *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/core *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/core *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/core/font *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/core/font *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/core/font/detail *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/core/font/detail *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/core/gui *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/core/gui *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/core/gui/detail *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/core/gui/detail *.h *.inl)

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

# set include directories
include_directories(
    ${SCM_ROOT_DIR}/scm_core/src
    # fixme, all gl dependent stuff has to go to scm_gl
    #${SCM_ROOT_DIR}/scm_gl/src
)
# set include directories for windows external dependencies
if (WIN32)
    include_directories(
        ${GLOBAL_EXT_DIR}/inc
        ${GLOBAL_EXT_DIR}/inc/boost
        ${GLOBAL_EXT_DIR}/inc/freetype
    )
elseif (UNIX)
    include_directories(
        /opt/boost/latest/inc
        /usr/include/freetype2
    )
endif (WIN32)

# set library directories
link_directories(
)
# set library directories for windows external dependencies
if (WIN32)
    link_directories(
        ${GLOBAL_EXT_DIR}/lib
    )
elseif (UNIX)
    link_directories(
        /opt/boost/latest/lib
    )
endif (WIN32)

if (WIN32)
    # add preprocessor defines
    add_definitions(-DSCM_BUILD_LIBRARY_core=\(\))
ENDIF (WIN32)

# add/create library
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

# link libraries
if (WIN32)
    target_link_libraries(${PROJECT_NAME}
        freetype2
        optimized libboost_program_options-${SCHISM_BOOST_LIB_POSTFIX_REL}  debug libboost_program_options-${SCHISM_BOOST_LIB_POSTFIX_DBG}
        optimized libboost_thread-${SCHISM_BOOST_LIB_POSTFIX_REL}           debug libboost_thread-${SCHISM_BOOST_LIB_POSTFIX_DBG}
        optimized libboost_date_time-${SCHISM_BOOST_LIB_POSTFIX_REL}        debug libboost_date_time-${SCHISM_BOOST_LIB_POSTFIX_DBG}
        optimized libboost_filesystem-${SCHISM_BOOST_LIB_POSTFIX_REL}       debug libboost_filesystem-${SCHISM_BOOST_LIB_POSTFIX_DBG}
        optimized libboost_system-${SCHISM_BOOST_LIB_POSTFIX_REL}           debug libboost_system-${SCHISM_BOOST_LIB_POSTFIX_DBG}
    )
endif (WIN32)

if (UNIX)
    target_link_libraries(${PROJECT_NAME}
        boost_date_time
        rt
        freetype
        boost_signals-${SCHISM_BOOST_LIB_POSTFIX_REL}
        boost_date_time-${SCHISM_BOOST_LIB_POSTFIX_REL}
        boost_filesystem-${SCHISM_BOOST_LIB_POSTFIX_REL}
    )
endif (UNIX)
