
project(scm_data)

include(schism_macros)
include(schism_boost)
include(schism_glsl)

set(SRC_DIR                 "${SCM_ROOT_DIR}/src/scm/data")
set(SCM_PROJECT_SOURCE_DIR  ${SRC_DIR})

set(LIBRARY_OUTPUT_PATH                         ${SCHISM_LIBRARY_DIR})

# source files
scm_project_files(SOURCE_FILES      ${SRC_DIR} *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR} *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/analysis *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/analysis *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/analysis/histogram *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/analysis/histogram *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/analysis/transfer_function *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/analysis/transfer_function *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/analysis/processing *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/analysis/processing *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/geometry *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/geometry *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/geometry/scm_geom *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/geometry/scm_geom *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/geometry/wavefront_obj *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/geometry/wavefront_obj *.h *.inl)

scm_project_files(SOURCE_FILES      ${SRC_DIR}/volume *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/volume *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/volume/voxel_geo_vol *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/volume/voxel_geo_vol *.h *.inl)
scm_project_files(SOURCE_FILES      ${SRC_DIR}/volume/scm_vol *.cpp)
scm_project_files(HEADER_FILES      ${SRC_DIR}/volume/scm_vol *.h *.inl)

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
    add_definitions(-DSCM_BUILD_LIBRARY_data=\(\))
endif (WIN32)

# add/create library
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

# link libraries
target_link_libraries(${PROJECT_NAME}
    scm_core
)
if (WIN32)
    target_link_libraries(${PROJECT_NAME}
        optimized libboost_filesystem-${SCHISM_BOOST_LIB_POSTFIX_REL}   debug libboost_filesystem-${SCHISM_BOOST_LIB_POSTFIX_DBG}
        optimized libboost_system-${SCHISM_BOOST_LIB_POSTFIX_REL}       debug libboost_system-${SCHISM_BOOST_LIB_POSTFIX_DBG}
    )
endif (WIN32)
if (UNIX)
    target_link_libraries(${PROJECT_NAME}
        boost_filesystem-${SCHISM_BOOST_LIB_POSTFIX_REL}
    )
endif (UNIX)

add_dependencies(${PROJECT_NAME}
    scm_core
)
