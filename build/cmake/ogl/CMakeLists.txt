
project(scm_ogl)

include(schism_macros)
include(schism_boost)
include(schism_glsl)

set(SRC_DIR                 "${SCM_ROOT_DIR}/src/scm/ogl")
set(SCM_PROJECT_SOURCE_DIR  ${SRC_DIR})

set(LIBRARY_OUTPUT_PATH                         ${SCHISM_LIBRARY_DIR})

# source files
scm_project_files(SOURCE_FILES      ${SRC_DIR} *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR} *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/buffer_object *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/buffer_object *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/font *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/font *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/framebuffer_object *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/framebuffer_object *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gui *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gui *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/manipulators *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/manipulators *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/primitives *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/primitives *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/shader_objects *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/shader_objects *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/system *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/system *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/textures *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/textures *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/time *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/time *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/utilities *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/utilities *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/vertexbuffer_object *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/vertexbuffer_object *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/math *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/math *.h *.inl)

# include header and inline files in source files for visual studio projects
if (WIN32)
    if (MSVC)
        set (SOURCE_FILES ${SOURCE_FILES} ${HEADER_FILES})
    endif (MSVC)
endif (WIN32)

# set include and lib directories
include_directories(
    ${SCM_ROOT_DIR}/src
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
