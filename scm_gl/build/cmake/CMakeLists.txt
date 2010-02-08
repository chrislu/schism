
project(scm_gl)

include(schism_macros)
include(schism_boost)
#include(schism_glsl)

set(SRC_DIR                         "${scm_gl_SOURCE_DIR}/../../src/scm")
set(SCM_PROJECT_SOURCE_DIR          ${SRC_DIR})

set(LIBRARY_OUTPUT_PATH                         ${SCHISM_LIBRARY_DIR})

# source files
scm_project_files(SOURCE_FILES      ${SRC_DIR} *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR} *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl/buffer_object *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl/buffer_object *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl/font *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl/font *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl/framebuffer_object *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl/framebuffer_object *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl/graphics_device *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl/graphics_device *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl/graphics_device/opengl3 *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl/graphics_device/opengl3 *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl/graphics_device/opengl3/detail *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl/graphics_device/opengl3/detail *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl/graphics_device/direct3d11 *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl/graphics_device/direct3d11 *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl/gui *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl/gui *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl/manipulators *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl/manipulators *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl/module *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl/module *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl/primitives *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl/primitives *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl/render_context *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl/render_context *.h *.inl)
#scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl/render_context/detail *.cpp)
#scm_project_files(HEADER_FILES      ${SRC_DIR}/gl/render_context/detail *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl/shader_objects *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl/shader_objects *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl/textures *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl/textures *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl/time *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl/time *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl/utilities *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl/utilities *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl/vertexbuffer_object *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl/vertexbuffer_object *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl/math *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl/math *.h *.inl)

# include header and inline files in source files for visual studio projects
if (WIN32)
    if (MSVC)
        set (SOURCE_FILES ${SOURCE_FILES} ${HEADER_FILES})
    endif (MSVC)
endif (WIN32)

# set include and lib directories
include_directories(
    ${SCM_ROOT_DIR}/scm_core/src
    ${SCM_ROOT_DIR}/scm_gl/src
)
# set include directories for windows external dependencies
if (WIN32)
    include_directories(
        ${GLOBAL_EXT_DIR}/inc
        ${GLOBAL_EXT_DIR}/inc/boost
    )
elseif (UNIX)
    include_directories(
        /opt/boost/latest/inc
    )
endif (WIN32)

# set library directories
link_directories(
    ${SCM_LIB_DIR}/${SCHISM_PLATFORM}
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
    add_definitions(-DSCM_BUILD_LIBRARY_ogl=\(\))
endif (WIN32)

# add/create library
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

# link libraries
target_link_libraries(${PROJECT_NAME}
    scm_core
)
if (WIN32)
    target_link_libraries(${PROJECT_NAME}
        glew32
        glut32
        opengl32
        optimized libboost_regex-${SCHISM_BOOST_LIB_POSTFIX_REL}           debug libboost_regex-${SCHISM_BOOST_LIB_POSTFIX_DBG}
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
