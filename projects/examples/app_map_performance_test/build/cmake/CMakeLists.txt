
PROJECT(app_map_performance_test)

include(schism_project)
include(schism_boost)
include(schism_macros)

# source files
scm_project_files(SOURCE_FILES      ${SRC_DIR} *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR} *.h *.inl)

# include header and inline files in source files for visual studio projects
if (WIN32)
    if (MSVC)
        set (SOURCE_FILES ${SOURCE_FILES} ${HEADER_FILES} ${SHADER_FILES})
    endif (MSVC)
endif (WIN32)

# set include directories
include_directories(
    ${SRC_DIR}
    ${SCM_ROOT_DIR}/scm_core/src
    ${SCM_BOOST_INC_DIR}
)

# set library directories
link_directories(
    ${SCM_LIB_DIR}/${SCHISM_PLATFORM}
    ${SCM_BOOST_LIB_DIR}
    ${GLOBAL_EXT_DIR}/lib
)

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
