
project(scm_gl_core)

include(schism_macros)
include(schism_boost)

set(SRC_DIR                         "${scm_gl_core_SOURCE_DIR}/../../src/scm")
set(SCM_PROJECT_SOURCE_DIR          ${SRC_DIR})

set(LIBRARY_OUTPUT_PATH             ${SCHISM_LIBRARY_DIR})

# source files
scm_project_files(SOURCE_FILES      ${SRC_DIR} *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR} *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/manipulators *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/manipulators *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/math *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/math *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/opengl *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/opengl *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/opengl/detail *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/opengl/detail *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/opengl/util *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/opengl/util *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/primitives *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/primitives *.h *.inl)

# include header and inline files in source files for visual studio projects
if (WIN32)
    if (MSVC)
        set (SOURCE_FILES ${SOURCE_FILES} ${HEADER_FILES})
    endif (MSVC)
endif (WIN32)


# set include and lib directories
scm_project_include_directories(ALL   ${SCM_ROOT_DIR}/scm_core/src
                                      ${SCM_ROOT_DIR}/scm_gl_core/src)
scm_project_include_directories(WIN32 ${GLOBAL_EXT_DIR}/inc
                                      ${GLOBAL_EXT_DIR}/inc/boost)
scm_project_include_directories(UNIX  /opt/boost/latest/inc)

# set library directories
scm_project_link_directories(ALL   ${SCM_LIB_DIR}/${SCHISM_PLATFORM})
scm_project_link_directories(WIN32 ${GLOBAL_EXT_DIR}/lib)
scm_project_link_directories(UNIX  /opt/boost/latest/lib)

if (WIN32)
    # add preprocessor defines
    add_definitions(-DSCM_BUILD_LIBRARY_gl_core=\(\))
endif (WIN32)

# add/create library
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

# link libraries
target_link_libraries(${PROJECT_NAME}
    scm_core
)
if (WIN32)
    target_link_libraries(${PROJECT_NAME}
        opengl32
        optimized libboost_regex-${SCHISM_BOOST_LIB_POSTFIX_REL}        debug libboost_regex-${SCHISM_BOOST_LIB_POSTFIX_DBG}
        optimized libboost_wave-${SCHISM_BOOST_LIB_POSTFIX_REL}         debug libboost_wave-${SCHISM_BOOST_LIB_POSTFIX_DBG}
    )
endif (WIN32)
if (UNIX)
    target_link_libraries(${PROJECT_NAME}
        GLEW
        glut
    )
endif (UNIX)

add_dependencies(${PROJECT_NAME}
    scm_core
)
