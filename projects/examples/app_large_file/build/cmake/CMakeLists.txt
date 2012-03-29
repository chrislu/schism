
# Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
# Distributed under the Modified BSD License, see license.txt.

PROJECT(app_large_file)

include(schism_project)
include(schism_boost)
include(schism_macros)

# source files
scm_project_files(SOURCE_FILES      ${SRC_DIR} *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR} *.h *.inl)

scm_project_files(SHADER_FILES      ${SRC_DIR}/shaders *.glsl *.glslf *.glslv *.glslg)

# include header and inline files in source files for visual studio projects
if (WIN32)
    if (MSVC)
        set (SOURCE_FILES ${SOURCE_FILES} ${HEADER_FILES} ${SHADER_FILES})
    endif (MSVC)
endif (WIN32)

# set include and lib directories
scm_project_include_directories(ALL   ${SRC_DIR}
                                      ${SCM_ROOT_DIR}/scm_core/src
                                      ${SCM_BOOST_INC_DIR})
scm_project_include_directories(WIN32 ${GLOBAL_EXT_DIR}/inc)
#scm_project_include_directories(UNIX  )

scm_project_link_directories(ALL   ${SCM_LIB_DIR}/${SCHISM_PLATFORM}
                                   ${SCM_BOOST_LIB_DIR})
scm_project_link_directories(WIN32 ${GLOBAL_EXT_DIR}/lib)
#scm_project_link_directories(UNIX  )

# add/create library
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# link libraries
scm_link_libraries(ALL
    general scm_core
)
#scm_link_libraries(WIN32 XXX)
#scm_link_libraries(UNIX  XXX)
scm_copy_schism_libraries()


add_dependencies(${PROJECT_NAME}
    scm_core
)
