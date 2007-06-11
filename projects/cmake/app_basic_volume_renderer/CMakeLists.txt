
PROJECT(app_basic_volume_renderer)

SET(SRC_DIR "${GLOBAL_SRC_DIR}/app_basic_volume_renderer")

SET(EXECUTABLE_OUTPUT_PATH     ${SCHISM_EXECUTABLE_DIR})

# source files
FILE(GLOB SOURCE_FILES                          ${SRC_DIR}/*.cpp)
FILE(GLOB HEADER_FILES                          ${SRC_DIR}/*.h)
FILE(GLOB INLINE_FILES                          ${SRC_DIR}/*.inl)

FILE(GLOB SOURCE_FILES_VOLUME_RENDERER          ${SRC_DIR}/volume_renderer/*.cpp)
FILE(GLOB HEADER_FILES_VOLUME_RENDERER          ${SRC_DIR}/volume_renderer/*.h)
FILE(GLOB INLINE_FILES_VOLUME_RENDERER          ${SRC_DIR}/volume_renderer/*.inl)

FILE(GLOB SHADER_FILES_VOLUME_RENDERER          ${SRC_DIR}/volume_renderer/shader/*.glsl)

FILE(GLOB SHADER_FILES                          ${SRC_DIR}/shader/*.glsl)

# define source groups for visual studio project files
SET(SOURCE_GROUP_DELIMITER "/")

SOURCE_GROUP(source_files                                       FILES ${SOURCE_FILES}                           ${HEADER_FILES}                         ${INLINE_FILES})
SOURCE_GROUP(source_files/shader                                FILES ${SHADER_FILES})
SOURCE_GROUP(source_files/volume_renderer                       FILES ${SOURCE_FILES_VOLUME_RENDERER}           ${HEADER_FILES_VOLUME_RENDERER}         ${INLINE_FILES_VOLUME_RENDERER})
SOURCE_GROUP(source_files/volume_renderer/shader                FILES ${SHADER_FILES_VOLUME_RENDERER})

# combine source file lists
SET(SOURCE_FILES
    ${SOURCE_FILES}
    ${SOURCE_FILES_VOLUME_RENDERER}
)

SET(HEADER_FILES
    ${HEADER_FILES}
    ${HEADER_FILES_VOLUME_RENDERER}
)

SET(INLINE_FILES
    ${INLINE_FILES}
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

# add/create library
ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCE_FILES})

# link libraries
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
    scm_ogl
    scm_core
    scm_data
)
IF (WIN32)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}
        glew32
        glut32
        optimized boost_filesystem-vc80-mt-${SCHISM_BOOST_VERSION}
        debug boost_filesystem-vc80-mt-gd-${SCHISM_BOOST_VERSION}
        optimized libboost_program_options-vc80-mt-${SCHISM_BOOST_VERSION}
        debug libboost_program_options-vc80-mt-gd-${SCHISM_BOOST_VERSION}
    )
ENDIF (WIN32)
IF (UNIX)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}
        GLEW
        glut
        boost_filesystem
        boost_program_options
    )
ENDIF (UNIX)

ADD_DEPENDENCIES(${PROJECT_NAME}
    scm_ogl
    scm_core
    scm_data
)
