
# Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
# Distributed under the Modified BSD License, see license.txt.

project(ex_volume_ray_cast_cuda)

include(schism_project)
include(schism_boost)
include(schism_cuda)
include(schism_macros)
include(schism_qt4)

# source files
scm_project_files(SOURCE_FILES      ${SRC_DIR} *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR} *.h *.inl)

scm_project_files(SOURCE_FILES_GUI  ${SRC_DIR}/application *.cpp)
scm_project_files(HEADER_FILES_GUI  ${SRC_DIR}/application *.h *.inl)
scm_project_files(SOURCE_FILES_GUI  ${SRC_DIR}/application/volume_gui *.cpp)
scm_project_files(HEADER_FILES_GUI  ${SRC_DIR}/application/volume_gui *.h *.inl)

scm_project_files(SOURCE_FILES_GUI  ${SRC_DIR}/gui_support *.cpp)
scm_project_files(HEADER_FILES_GUI  ${SRC_DIR}/gui_support *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/renderer *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/renderer *.h *.inl)
scm_project_files(SHADER_FILES      ${SRC_DIR}/renderer/shader *.glsl *.glslh *.glslf *.glslv *glslg *.glsltc *.glslte)
scm_project_files(KERNEL_FILES      ${SRC_DIR}/renderer/kernel *.cl *.clh *.cu *.h)

scm_add_moc_command(SOURCE_FILES_GUI ${HEADER_FILES_GUI})
list(APPEND HEADER_FILES ${HEADER_FILES_GUI})
list(APPEND SOURCE_FILES ${SOURCE_FILES_GUI})

# include header and inline files in source files for visual studio projects
if (WIN32)
    if (MSVC)
        set (SOURCE_FILES ${SOURCE_FILES} ${HEADER_FILES} ${SHADER_FILES} ${KERNEL_FILES})
    endif (MSVC)
endif (WIN32)

# set include and lib directories
scm_project_include_directories(ALL   ${SRC_DIR}
                                      ${SCM_ROOT_DIR}/scm_core/src
                                      ${SCM_ROOT_DIR}/scm_data/src
                                      ${SCM_ROOT_DIR}/scm_cl_core/src
                                      ${SCM_ROOT_DIR}/scm_gl_core/src
                                      ${SCM_ROOT_DIR}/scm_gl_util/src
                                      ${SCM_ROOT_DIR}/scm_input/src
                                      ${SCM_BOOST_INC_DIR}
                                      ${SCM_QT_INC_DIR})

scm_project_include_directories(WIN32 ${GLOBAL_EXT_DIR}/inc
	                                  ${GLOBAL_EXT_DIR}/inc/cuda
                                      ${GLOBAL_EXT_DIR}/inc/freeimage)
scm_project_include_directories(UNIX  /opt/OpenCL/include)

#scm_cuda_project_include_directories( ${GLOBAL_EXT_DIR}/inc
#                                      ${GLOBAL_EXT_DIR}/inc/cuda)
#message(${PROJECT_NAME} ${SCM_CUDA_NVCC_INCLUDE_DIRECTORIES})

# set library directories
scm_project_link_directories(ALL      ${SCM_LIB_DIR}/${SCHISM_PLATFORM}
                                      ${SCM_BOOST_LIB_DIR}
                                      ${SCM_QT_LIB_DIR})

#scm_project_link_directories(WIN32    ${GLOBAL_EXT_DIR}/lib)
#scm_project_link_directories(UNIX     )

# add/create library
scm_cuda_add_nvcc_prepass(${SOURCE_FILES})
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# link libraries
scm_link_libraries(ALL
    general scm_core
    general scm_cl_core
    general scm_gl_core
    general scm_gl_util
    general scm_data

	general cuda
	general cudart
)
scm_link_libraries(WIN32
    optimized libboost_filesystem-${SCM_BOOST_MT_REL}       debug libboost_filesystem-${SCM_BOOST_MT_DBG}
    optimized libboost_program_options-${SCM_BOOST_MT_REL}  debug libboost_program_options-${SCM_BOOST_MT_DBG}
    optimized QtCore4       debug QtCored4
    optimized QtGui4        debug QtGuid4
    general FreeImage
    general FreeImagePlus
)
scm_link_libraries(UNIX
    general boost_filesystem${SCM_BOOST_MT_REL}
    general boost_program_options${SCM_BOOST_MT_REL}
    general QtCore
    general QtGui
)
scm_copy_schism_libraries()
scm_copy_shared_libraries(WIN32 cudart64_41_28.dll)

add_dependencies(${PROJECT_NAME}
    scm_core
    scm_cl_core
    scm_gl_core
    scm_gl_util
    scm_data
)
