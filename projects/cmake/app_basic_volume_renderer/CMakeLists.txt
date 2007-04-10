
PROJECT(app_basic_volume_renderer)

SET(SRC_DIR "${GLOBAL_SRC_DIR}/app_basic_volume_renderer")

SET(EXECUTABLE_OUTPUT_PATH     ${SCHISM_EXECUTABLE_DIR})

# source files
FILE(GLOB SOURCE_FILES                          ${SRC_DIR}/*.cpp)
FILE(GLOB HEADER_FILES                          ${SRC_DIR}/*.h)
FILE(GLOB INLINE_FILES                          ${SRC_DIR}/*.inl)

FILE(GLOB SOURCE_FILES_DATA_ANALYSIS            ${SRC_DIR}/data_analysis/*.cpp)
FILE(GLOB HEADER_FILES_DATA_ANALYSIS            ${SRC_DIR}/data_analysis/*.h)
FILE(GLOB INLINE_FILES_DATA_ANALYSIS            ${SRC_DIR}/data_analysis/*.inl)

FILE(GLOB SOURCE_FILES_DATA_ANALYSIS_HISTOGRAM  ${SRC_DIR}/data_analysis/histogram/*.cpp)
FILE(GLOB HEADER_FILES_DATA_ANALYSIS_HISTOGRAM  ${SRC_DIR}/data_analysis/histogram/*.h)
FILE(GLOB INLINE_FILES_DATA_ANALYSIS_HISTOGRAM  ${SRC_DIR}/data_analysis/histogram/*.inl)

FILE(GLOB SOURCE_FILES_DATA_ANALYSIS_TRAFU      ${SRC_DIR}/data_analysis/transfer_function/*.cpp)
FILE(GLOB HEADER_FILES_DATA_ANALYSIS_TRAFU      ${SRC_DIR}/data_analysis/transfer_function/*.h)
FILE(GLOB INLINE_FILES_DATA_ANALYSIS_TRAFU      ${SRC_DIR}/data_analysis/transfer_function/*.inl)

FILE(GLOB SOURCE_FILES_SYSTEM                   ${SRC_DIR}/system/*.cpp)
FILE(GLOB HEADER_FILES_SYSTEM                   ${SRC_DIR}/system/*.h)
FILE(GLOB INLINE_FILES_SYSTEM                   ${SRC_DIR}/system/*.inl)

FILE(GLOB SOURCE_FILES_VOLUME_LOADER            ${SRC_DIR}/volume_loader/*.cpp)
FILE(GLOB HEADER_FILES_VOLUME_LOADER            ${SRC_DIR}/volume_loader/*.h)
FILE(GLOB INLINE_FILES_VOLUME_LOADER            ${SRC_DIR}/volume_loader/*.inl)

FILE(GLOB SOURCE_FILES_VOLUME_LOADER_VGEO       ${SRC_DIR}/volume_loader/voxel_geo_vol/*.cpp)
FILE(GLOB HEADER_FILES_VOLUME_LOADER_VGEO       ${SRC_DIR}/volume_loader/voxel_geo_vol/*.h)
FILE(GLOB INLINE_FILES_VOLUME_LOADER_VGEO       ${SRC_DIR}/volume_loader/voxel_geo_vol/*.inl)

FILE(GLOB SOURCE_FILES_VOLUME_RENDERER          ${SRC_DIR}/volume_renderer/*.cpp)
FILE(GLOB HEADER_FILES_VOLUME_RENDERER          ${SRC_DIR}/volume_renderer/*.h)
FILE(GLOB INLINE_FILES_VOLUME_RENDERER          ${SRC_DIR}/volume_renderer/*.inl)

FILE(GLOB SHADER_FILES_VOLUME_RENDERER          ${SRC_DIR}/volume_renderer/shader/*.glsl)

FILE(GLOB SHADER_FILES                          ${SRC_DIR}/shader/*.glsl)

# define source groups for visual studio project files
SET(SOURCE_GROUP_DELIMITER "/")

SOURCE_GROUP(source_files                                       FILES ${SOURCE_FILES}                           ${HEADER_FILES}                         ${INLINE_FILES})
SOURCE_GROUP(source_files/data_analysis                         FILES ${SOURCE_FILES_DATA_ANALYSIS}             ${HEADER_FILES_DATA_ANALYSIS}           ${INLINE_FILES_DATA_ANALYSIS})
SOURCE_GROUP(source_files/data_analysis/histogram               FILES ${SOURCE_FILES_DATA_ANALYSIS_HISTOGRAM}   ${HEADER_FILES_DATA_ANALYSIS_HISTOGRAM} ${INLINE_FILES_DATA_ANALYSIS_HISTOGRAM})
SOURCE_GROUP(source_files/data_analysis/transfer_function       FILES ${SOURCE_FILES_DATA_ANALYSIS_TRAFU}       ${HEADER_FILES_DATA_ANALYSIS_TRAFU}     ${INLINE_FILES_DATA_ANALYSIS_TRAFU})
SOURCE_GROUP(source_files/shader                                FILES ${SHADER_FILES})
SOURCE_GROUP(source_files/system                                FILES ${SOURCE_FILES_SYSTEM}                    ${HEADER_FILES_SYSTEM}                  ${INLINE_FILES_SYSTEM})
SOURCE_GROUP(source_files/volume_loader                         FILES ${SOURCE_FILES_VOLUME_LOADER}             ${HEADER_FILES_VOLUME_LOADER}           ${INLINE_FILES_VOLUME_LOADER})
SOURCE_GROUP(source_files/volume_loader/voxel_geo_vol           FILES ${SOURCE_FILES_VOLUME_LOADER_VGEO}        ${HEADER_FILES_VOLUME_LOADER_VGEO}      ${INLINE_FILES_VOLUME_LOADER_VGEO})
SOURCE_GROUP(source_files/volume_renderer                       FILES ${SOURCE_FILES_VOLUME_RENDERER}           ${HEADER_FILES_VOLUME_RENDERER}         ${INLINE_FILES_VOLUME_RENDERER})
SOURCE_GROUP(source_files/volume_renderer/shader                FILES ${SHADER_FILES_VOLUME_RENDERER})

# combine source file lists
SET(SOURCE_FILES
    ${SOURCE_FILES}
    ${SOURCE_FILES_DATA_ANALYSIS}
    ${SOURCE_FILES_DATA_ANALYSIS_HISTOGRAM}
    ${SOURCE_FILES_DATA_ANALYSIS_TRAFU}
    ${SOURCE_FILES_SYSTEM}
    ${SOURCE_FILES_VOLUME_LOADER}
    ${SOURCE_FILES_VOLUME_LOADER_VGEO}
    ${SOURCE_FILES_VOLUME_RENDERER}
)

SET(HEADER_FILES
    ${HEADER_FILES}
    ${HEADER_FILES_DATA_ANALYSIS}
    ${HEADER_FILES_DATA_ANALYSIS_HISTOGRAM}
    ${HEADER_FILES_DATA_ANALYSIS_TRAFU}
    ${HEADER_FILES_SYSTEM}
    ${HEADER_FILES_VOLUME_LOADER}
    ${HEADER_FILES_VOLUME_LOADER_VGEO}
    ${HEADER_FILES_VOLUME_RENDERER}
)

SET(INLINE_FILES
    ${INLINE_FILES}
    ${INLINE_FILES_DATA_ANALYSIS}
    ${INLINE_FILES_DATA_ANALYSIS_HISTOGRAM}
    ${INLINE_FILES_DATA_ANALYSIS_TRAFU}
    ${INLINE_FILES_SYSTEM}
    ${INLINE_FILES_VOLUME_LOADER}
    ${INLINE_FILES_VOLUME_LOADER_VGEO}
    ${INLINE_FILES_VOLUME_RENDERER}
)

SET(SHADER_FILES
    ${SHADER_FILES}
    ${SHADER_FILES_VOLUME_RENDERER}
)

# include header and inline files in source files for visual studio projects
IF (WIN32)
    IF (MSVC)
        SET (SOURCE_FILES ${SOURCE_FILES} ${HEADER_FILES} ${INLINE_FILES} ${SHADER_FILES})
    ENDIF (MSVC)
ENDIF (WIN32)

# set include directories
INCLUDE_DIRECTORIES(
    ${GLOBAL_SRC_DIR}/app_basic_volume_renderer
    ${GLOBAL_SRC_DIR}/lib_ogl
    ${GLOBAL_SRC_DIR}/lib_scm_core
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
        ${GLOBAL_EXT_DIR}/lib/${SCHISM_PLATFORM}
    )
ENDIF (WIN32)

# add/create library
ADD_EXECUTABLE(app_basic_volume_renderer ${SOURCE_FILES})

# link libraries
TARGET_LINK_LIBRARIES(app_basic_volume_renderer
    lib_ogl
    lib_scm_core
)
IF (WIN32)
    TARGET_LINK_LIBRARIES(app_basic_volume_renderer
        glew32
        glut32
        optimized boost_filesystem-vc80-mt-${SCHISM_BOOST_VERSION}
        debug boost_filesystem-vc80-mt-gd-${SCHISM_BOOST_VERSION}
    )
ENDIF (WIN32)
IF (UNIX)
    TARGET_LINK_LIBRARIES(app_basic_volume_renderer
        GLEW
        glut
        boost_filesystem
    )
ENDIF (UNIX)

ADD_DEPENDENCIES(app_basic_volume_renderer
    lib_ogl
    lib_scm_core
)


