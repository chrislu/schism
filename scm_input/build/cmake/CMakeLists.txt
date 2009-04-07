
project(scm_input)

include(schism_macros)
include(schism_boost)

set(SRC_DIR                 "${scm_input_SOURCE_DIR}/../../src/scm")
set(SCM_PROJECT_SOURCE_DIR  ${SRC_DIR})

set(LIBRARY_OUTPUT_PATH                         ${SCHISM_LIBRARY_DIR})

# source files
scm_project_files(SOURCE_FILES      ${SRC_DIR} *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR} *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/input *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR} *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/input/tracking *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/input/tracking *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/input/tracking/detail *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/input/tracking/detail *.h *.inl)

# include header and inline files in source files for visual studio projects
if (WIN32)
    if (MSVC)
        set (SOURCE_FILES ${SOURCE_FILES} ${HEADER_FILES})
    endif (MSVC)
endif (WIN32)

# set include and lib directories
include_directories(
    ${SCM_ROOT_DIR}/scm_core/src
    ${SCM_ROOT_DIR}/scm_input/src
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
    add_definitions(-DSCM_BUILD_LIBRARY_input=\(\))
endif (WIN32)

# add/create library
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

# link libraries
target_link_libraries(${PROJECT_NAME}
    scm_core
)
if (WIN32)
    target_link_libraries(${PROJECT_NAME}
        ws2_32
    )
endif (WIN32)
if (UNIX)
    target_link_libraries(${PROJECT_NAME}
    )
endif (UNIX)

add_dependencies(${PROJECT_NAME}
    scm_core
)
