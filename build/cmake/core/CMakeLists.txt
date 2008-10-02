
project(scm_core)

include(schism_macros)
include(schism_glsl)

set(SRC_DIR                 "${SCHISM_SRC_DIR}/scm/core")
set(SCM_PROJECT_SOURCE_DIR  ${SRC_DIR})

SET(LIBRARY_OUTPUT_PATH             ${SCHISM_LIBRARY_DIR})

scm_project_files(SOURCE_FILES      ${SRC_DIR} *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR} *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/schism/scm *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/schism/scm *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/console *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/console *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/console/detail *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/console/detail *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/root *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/root *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/root/detail *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/root/detail *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/exception *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/exception *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/font *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/font *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/font/detail *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/font/detail *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gui *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gui *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/gui/detail *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gui/detail *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/io *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/io *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/io/detail *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/io/detail *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/math *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/math *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/math/detail *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/math/detail *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/platform *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/platform *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/platform/graphics *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/platform/graphics *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/platform/graphics/detail *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/platform/graphics/detail *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/resource *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/resource *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/resource/detail *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/resource/detail *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/script *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/script *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/script/detail *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/script/detail *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/time *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/time *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/time/detail *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/time/detail *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/utilities *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/utilities *.h *.inl)

# include header and inline files in source files for visual studio projects
if (WIN32)
    if (MSVC)
        set (SOURCE_FILES ${SOURCE_FILES} ${HEADER_FILES})
    endif (MSVC)
endif (WIN32)

# set include directories
include_directories(
    ${SCHISM_SRC_DIR}
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
        optimized libboost_signals-${SCHISM_BOOST_LIB_POSTFIX_REL}      debug libboost_signals-${SCHISM_BOOST_LIB_POSTFIX_DBG}
        optimized libboost_date_time-${SCHISM_BOOST_LIB_POSTFIX_REL}    debug libboost_date_time-${SCHISM_BOOST_LIB_POSTFIX_DBG}
        optimized libboost_filesystem-${SCHISM_BOOST_LIB_POSTFIX_REL}   debug libboost_filesystem-${SCHISM_BOOST_LIB_POSTFIX_DBG}
        optimized libboost_system-${SCHISM_BOOST_LIB_POSTFIX_REL}       debug libboost_system-${SCHISM_BOOST_LIB_POSTFIX_DBG}
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
