
PROJECT(app_volume_converter)

SET(SRC_DIR "${GLOBAL_SRC_DIR}/app_volume_converter")

SET(EXECUTABLE_OUTPUT_PATH     ${SCHISM_EXECUTABLE_DIR})

# source files
FILE(GLOB SOURCE_FILES                          ${SRC_DIR}/*.cpp)
FILE(GLOB HEADER_FILES                          ${SRC_DIR}/*.h)
FILE(GLOB INLINE_FILES                          ${SRC_DIR}/*.inl)

# define source groups for visual studio project files
SET(SOURCE_GROUP_DELIMITER "/")

SOURCE_GROUP(source_files                                       FILES ${SOURCE_FILES}                           ${HEADER_FILES}                         ${INLINE_FILES})

# combine source file lists
SET(SOURCE_FILES
    ${SOURCE_FILES}
)

SET(HEADER_FILES
    ${HEADER_FILES}
)

SET(INLINE_FILES
    ${INLINE_FILES}
)

SET(SHADER_FILES
    ${SHADER_FILES}
)

# include header and inline files in source files for visual studio projects
IF (WIN32)
    IF (MSVC)
        SET (SOURCE_FILES ${SOURCE_FILES} ${HEADER_FILES} ${INLINE_FILES} ${SHADER_FILES})
    ENDIF (MSVC)
ENDIF (WIN32)

# set include directories
INCLUDE_DIRECTORIES(
    ${GLOBAL_SRC_DIR}/app_volume_converter
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
    scm_core
    scm_data
)
IF (WIN32)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}
        optimized boost_filesystem-vc80-mt-${SCHISM_BOOST_VERSION}
        debug boost_filesystem-vc80-mt-gd-${SCHISM_BOOST_VERSION}
        optimized libboost_program_options-vc80-mt-${SCHISM_BOOST_VERSION}
        debug libboost_program_options-vc80-mt-gd-${SCHISM_BOOST_VERSION}
    )
ENDIF (WIN32)
IF (UNIX)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}
        boost_filesystem
        boost_program_options
    )
ENDIF (UNIX)

ADD_DEPENDENCIES(${PROJECT_NAME}
    scm_core
    scm_data
)
