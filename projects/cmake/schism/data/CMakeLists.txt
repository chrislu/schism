
PROJECT(scm_data)

SET(SRC_DIR "${GLOBAL_SRC_DIR}/schism/scm/data")

SET(LIBRARY_OUTPUT_PATH                         ${SCHISM_LIBRARY_DIR})

# source files
FILE(GLOB SOURCE_FILES                          ${SRC_DIR}/*.cpp)
FILE(GLOB HEADER_FILES                          ${SRC_DIR}/*.h)
FILE(GLOB INLINE_FILES                          ${SRC_DIR}/*.inl)

FILE(GLOB SOURCE_FILES_ANALYSIS                 ${SRC_DIR}/analysis/*.cpp)
FILE(GLOB HEADER_FILES_ANALYSIS                 ${SRC_DIR}/analysis/*.h)
FILE(GLOB INLINE_FILES_ANALYSIS                 ${SRC_DIR}/analysis/*.inl)

FILE(GLOB SOURCE_FILES_ANALYSIS_HISTOGRAM		${SRC_DIR}/analysis/histogram/*.cpp)
FILE(GLOB HEADER_FILES_ANALYSIS_HISTOGRAM		${SRC_DIR}/analysis/histogram/*.h)
FILE(GLOB INLINE_FILES_ANALYSIS_HISTOGRAM		${SRC_DIR}/analysis/histogram/*.inl)

FILE(GLOB SOURCE_FILES_ANALYSIS_TRAFU			${SRC_DIR}/analysis/transfer_function/*.cpp)
FILE(GLOB HEADER_FILES_ANALYSIS_TRAFU			${SRC_DIR}/analysis/transfer_function/*.h)
FILE(GLOB INLINE_FILES_ANALYSIS_TRAFU			${SRC_DIR}/analysis/transfer_function/*.inl)

FILE(GLOB SOURCE_FILES_ANALYSIS_PROC			${SRC_DIR}/analysis/processing/*.cpp)
FILE(GLOB HEADER_FILES_ANALYSIS_PROC			${SRC_DIR}/analysis/processing/*.h)
FILE(GLOB INLINE_FILES_ANALYSIS_PROC			${SRC_DIR}/analysis/processing/*.inl)

FILE(GLOB SOURCE_FILES_GEOMETRY                 ${SRC_DIR}/geometry/*.cpp)
FILE(GLOB HEADER_FILES_GEOMETRY                 ${SRC_DIR}/geometry/*.h)
FILE(GLOB INLINE_FILES_GEOMETRY                 ${SRC_DIR}/geometry/*.inl)

FILE(GLOB SOURCE_FILES_GEOMETRY_SCMGEO          ${SRC_DIR}/geometry/scm_geom/*.cpp)
FILE(GLOB HEADER_FILES_GEOMETRY_SCMGEO          ${SRC_DIR}/geometry/scm_geom/*.h)
FILE(GLOB INLINE_FILES_GEOMETRY_SCMGEO          ${SRC_DIR}/geometry/scm_geom/*.inl)

FILE(GLOB SOURCE_FILES_GEOMETRY_WFOBJ           ${SRC_DIR}/geometry/wavefront_obj/*.cpp)
FILE(GLOB HEADER_FILES_GEOMETRY_WFOBJ           ${SRC_DIR}/geometry/wavefront_obj/*.h)
FILE(GLOB INLINE_FILES_GEOMETRY_WFOBJ           ${SRC_DIR}/geometry/wavefront_obj/*.inl)

FILE(GLOB SOURCE_FILES_VOLUME                   ${SRC_DIR}/volume/*.cpp)
FILE(GLOB HEADER_FILES_VOLUME                   ${SRC_DIR}/volume/*.h)
FILE(GLOB INLINE_FILES_VOLUME                   ${SRC_DIR}/volume/*.inl)

FILE(GLOB HEADER_FILES_VOLUME_VOXGEO            ${SRC_DIR}/volume/voxel_geo_vol/*.h)

FILE(GLOB SOURCE_FILES_VOLUME_SCM_VOL           ${SRC_DIR}/volume/scm_vol/*.cpp)
FILE(GLOB HEADER_FILES_VOLUME_SCM_VOL           ${SRC_DIR}/volume/scm_vol/*.h)

# define source grous
SET(SOURCE_GROUP_DELIMITER "/")

SOURCE_GROUP(source_files						        FILES ${SOURCE_FILES}					    ${HEADER_FILES}                     ${INLINE_FILES})
SOURCE_GROUP(source_files/analysis                      FILES ${SOURCE_FILES_ANALYSIS}              ${HEADER_FILES_ANALYSIS}            ${INLINE_FILES_ANALYSIS})
SOURCE_GROUP(source_files/analysis/histogram            FILES ${SOURCE_FILES_ANALYSIS_HISTOGRAM}    ${HEADER_FILES_ANALYSIS_HISTOGRAM}  ${INLINE_FILES_ANALYSIS_HISTOGRAM})
SOURCE_GROUP(source_files/analysis/transfer_function    FILES ${SOURCE_FILES_ANALYSIS_TRAFU}        ${HEADER_FILES_ANALYSIS_TRAFU}      ${INLINE_FILES_ANALYSIS_TRAFU})
SOURCE_GROUP(source_files/analysis/processing           FILES ${SOURCE_FILES_ANALYSIS_PROC}         ${HEADER_FILES_ANALYSIS_PROC}       ${INLINE_FILES_ANALYSIS_PROC})
SOURCE_GROUP(source_files/geometry                      FILES ${SOURCE_FILES_GEOMETRY}              ${HEADER_FILES_GEOMETRY}            ${INLINE_FILES_GEOMETRY})
SOURCE_GROUP(source_files/geometry/scm_geom             FILES ${SOURCE_FILES_GEOMETRY_SCMGEO}       ${HEADER_FILES_GEOMETRY_SCMGEO}     ${INLINE_FILES_GEOMETRY_SCMGEO})
SOURCE_GROUP(source_files/geometry/wavefront_obj        FILES ${SOURCE_FILES_GEOMETRY_WFOBJ}        ${HEADER_FILES_GEOMETRY_WFOBJ}      ${INLINE_FILES_GEOMETRY_WFOBJ})
SOURCE_GROUP(source_files/volume                        FILES ${SOURCE_FILES_VOLUME}                ${HEADER_FILES_VOLUME}              ${INLINE_FILES_VOLUME})
SOURCE_GROUP(source_files/volume/voxel_geo              FILES                                       ${HEADER_FILES_VOLUME_VOXGEO})
SOURCE_GROUP(source_files/volume/scm_vol                FILES ${SOURCE_FILES_VOLUME_SCM_VOL}        ${HEADER_FILES_VOLUME_SCM_VOL})

# combine source file lists
SET(SOURCE_FILES
    ${SOURCE_FILES}
    ${SOURCE_FILES_ANALYSIS}
    ${SOURCE_FILES_ANALYSIS_HISTOGRAM}
    ${SOURCE_FILES_ANALYSIS_TRAFU}
    ${SOURCE_FILES_ANALYSIS_PROC}
    ${SOURCE_FILES_GEOMETRY}
    ${SOURCE_FILES_GEOMETRY_SCMGEO}
    ${SOURCE_FILES_GEOMETRY_WFOBJ}
    ${SOURCE_FILES_VOLUME}
    ${SOURCE_FILES_VOLUME_SCM_VOL}
)

SET(HEADER_FILES
    ${HEADER_FILES}
    ${HEADER_FILES_ANALYSIS}
    ${HEADER_FILES_ANALYSIS_HISTOGRAM}
    ${HEADER_FILES_ANALYSIS_TRAFU}
    ${HEADER_FILES_ANALYSIS_PROC}
    ${HEADER_FILES_GEOMETRY}
    ${HEADER_FILES_GEOMETRY_SCMGEO}
    ${HEADER_FILES_GEOMETRY_WFOBJ}
    ${HEADER_FILES_VOLUME}
    ${HEADER_FILES_VOLUME_VOXGEO}
    ${HEADER_FILES_VOLUME_SCM_VOL}
)

SET(INLINE_FILES
    ${INLINE_FILES}
    ${INLINE_FILES_ANALYSIS}
    ${INLINE_FILES_ANALYSIS_HISTOGRAM}
    ${INLINE_FILES_ANALYSIS_TRAFU}
    ${INLINE_FILES_ANALYSIS_PROC}
    ${INLINE_FILES_GEOMETRY}
    ${INLINE_FILES_GEOMETRY_SCMGEO}
    ${INLINE_FILES_GEOMETRY_WFOBJ}
    ${INLINE_FILES_VOLUME}
)

# include header and inline files in source files for visual studio projects
IF (WIN32)
    IF (MSVC)
        SET (SOURCE_FILES ${SOURCE_FILES} ${HEADER_FILES} ${INLINE_FILES})
    ENDIF (MSVC)
ENDIF (WIN32)

# set include and lib directories
INCLUDE_DIRECTORIES(
    ${GLOBAL_SRC_DIR}/schism
)
# set include directories for windows external dependencies
IF (WIN32)
    INCLUDE_DIRECTORIES(
        ${GLOBAL_EXT_DIR}/inc
        ${GLOBAL_EXT_DIR}/inc/boost
    )
ENDIF (WIN32)

# set library directories
LINK_DIRECTORIES(
    ${GLOBAL_LIB_DIR}/${SCHISM_PLATFORM}
)
# set library directories for windows external dependencies
IF (WIN32)
    LINK_DIRECTORIES(
        ${GLOBAL_EXT_DIR}/lib
    )
ENDIF (WIN32)

IF (WIN32)
    # add preprocessor defines
    ADD_DEFINITIONS(-DSCM_BUILD_LIBRARY_data=\(\))
ENDIF (WIN32)

# add/create library
ADD_LIBRARY(${PROJECT_NAME} SHARED ${SOURCE_FILES})

# link libraries
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
    scm_core
)
IF (WIN32)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}
        optimized libboost_filesystem-${SCHISM_BOOST_LIB_POSTFIX_REL}   debug libboost_filesystem-${SCHISM_BOOST_LIB_POSTFIX_DBG}
    )
ENDIF (WIN32)
IF (UNIX)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}
    )
ENDIF (UNIX)

ADD_DEPENDENCIES(${PROJECT_NAME}
    scm_core
)
