
# Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
# Distributed under the Modified BSD License, see license.txt.

project(scm_gl_core)

option(SCHISM_OPT_BUILD_deprecated_classic_scm_gl  "Build deprecated classic scm_gl library (namespace scm::gl_classic"     OFF)

include(schism_macros)
include(schism_boost)

set(SRC_DIR                         "${scm_gl_core_SOURCE_DIR}/../../src/scm")
set(SCM_PROJECT_SOURCE_DIR          ${SRC_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${SCHISM_LIBRARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${SCHISM_LIBRARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${SCHISM_LIBRARY_DIR})

# source files
scm_project_files(SOURCE_FILES      ${SRC_DIR} *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR} *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/buffer_objects *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/buffer_objects *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/frame_buffer_objects *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/frame_buffer_objects *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/render_device *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/render_device *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/render_device/opengl *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/render_device/opengl *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/render_device/opengl/GL *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/render_device/opengl/GL *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/render_device/opengl/GLES2 *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/render_device/opengl/GLES2 *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/render_device/opengl/detail *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/render_device/opengl/detail *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/render_device/opengl/util *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/render_device/opengl/util *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/math *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/math *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/primitives *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/primitives *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/query_objects *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/query_objects *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/shader_objects *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/shader_objects *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/state_objects *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/state_objects *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/sync_objects *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/sync_objects *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/texture_objects *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/texture_objects *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/window_management *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/window_management *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/window_management/wm_win32 *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/window_management/wm_win32 *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/window_management/wm_win32/util *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/window_management/wm_win32/util *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/window_management/wm_x *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/window_management/wm_x *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_core/window_management/wm_x/util *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_core/window_management/wm_x/util *.h *.inl)

if (SCHISM_OPT_BUILD_deprecated_classic_scm_gl)
## gl_classic #####################################################################################
scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_classic *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_classic *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_classic/buffer_object *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_classic/buffer_object *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_classic/framebuffer_object *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_classic/framebuffer_object *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_classic/module *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_classic/module *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_classic/primitives *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_classic/primitives *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_classic/render_context *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_classic/render_context *.h *.inl)
#scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_classic/render_context/detail *.cpp)
#scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_classic/render_context/detail *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_classic/shader_objects *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_classic/shader_objects *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_classic/textures *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_classic/textures *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_classic/time *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_classic/time *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_classic/utilities *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_classic/utilities *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/gl_classic/vertexbuffer_object *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/gl_classic/vertexbuffer_object *.h *.inl)
## gl_classic #####################################################################################
endif (SCHISM_OPT_BUILD_deprecated_classic_scm_gl)

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
                                      ${SCM_BOOST_INC_DIR})
scm_project_include_directories(WIN32 ${GLOBAL_EXT_DIR}/inc)

# set library directories
scm_project_link_directories(ALL   ${SCM_LIB_DIR}/${SCHISM_PLATFORM}
                                   ${SCM_BOOST_LIB_DIR})
scm_project_link_directories(WIN32 ${GLOBAL_EXT_DIR}/lib)
scm_project_link_directories(UNIX     /opt/cuda/current/cuda/lib64
                                      /usr/lib/nvidia-current)

#scm_project_link_directories(UNIX  )

if (WIN32)
    # add preprocessor defines
    add_definitions(-DSCM_BUILD_LIBRARY_gl_core=\(\))
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
	general scm_core
    general scm_cl_core
)
scm_link_libraries(WIN32
    general opengl32
)
scm_link_libraries(UNIX
    general GL
)

if (SCHISM_OPT_BUILD_deprecated_classic_scm_gl)
	scm_link_libraries(WIN32
	    general glew32
	)
	scm_link_libraries(UNIX
	    general GLEW
	)
endif (SCHISM_OPT_BUILD_deprecated_classic_scm_gl)

add_dependencies(${PROJECT_NAME}
    scm_core
)

IF (${SCM_ENABLE_CUDA_CL_SUPPORT})
	add_dependencies(${PROJECT_NAME}
		scm_cl_core
	)
ENDIF (${SCM_ENABLE_CUDA_CL_SUPPORT})  


