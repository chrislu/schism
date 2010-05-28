
project(scm_gl_util)

include(schism_macros)
include(schism_boost)

set(SRC_DIR                         "${scm_gl_util_SOURCE_DIR}/../../src/scm")
set(SCM_PROJECT_SOURCE_DIR          ${SRC_DIR})

set(LIBRARY_OUTPUT_PATH             ${SCHISM_LIBRARY_DIR})

# source files
scm_project_files(SOURCE_FILES      ${SRC_DIR} *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR} *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_util *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_util *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_util/imaging *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_util/imaging *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_util/manipulators *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_util/manipulators *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_util/primitives *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_util/primitives *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_util/primitives/util *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_util/primitives/util *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_util/render_context *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_util/render_context *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_util/render_context/detail *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_util/render_context/detail *.h *.inl)

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
                                      ${SCM_ROOT_DIR}/scm_gl_core/src
                                      ${SCM_ROOT_DIR}/scm_gl_util/src)
scm_project_include_directories(WIN32 ${GLOBAL_EXT_DIR}/inc
                                      ${GLOBAL_EXT_DIR}/inc/boost)
scm_project_include_directories(UNIX  /opt/boost/latest/inc)

# set library directories
scm_project_link_directories(ALL   ${SCM_LIB_DIR}/${SCHISM_PLATFORM})
scm_project_link_directories(WIN32 ${GLOBAL_EXT_DIR}/lib)
scm_project_link_directories(UNIX  /opt/boost/latest/lib)

if (WIN32)
    # add preprocessor defines
    add_definitions(-DSCM_BUILD_LIBRARY_gl_util=\(\))
endif (WIN32)

# add/create library
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

# link libraries
target_link_libraries(${PROJECT_NAME}
    scm_core
    scm_gl_core
)
if (WIN32)
    target_link_libraries(${PROJECT_NAME}
        FreeImagePlus
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
    scm_gl_core
)
