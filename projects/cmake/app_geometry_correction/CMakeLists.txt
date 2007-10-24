
PROJECT(app_geometry_correction)

SET(SRC_DIR "${GLOBAL_SRC_DIR}/app_geometry_correction")

SET(EXECUTABLE_OUTPUT_PATH     ${SCHISM_EXECUTABLE_DIR})

# source files
FILE(GLOB SOURCE_FILES                  ${SRC_DIR}/*.cpp)
FILE(GLOB HEADER_FILES                  ${SRC_DIR}/*.h
                                        ${SRC_DIR}/*.inl)

FILE(GLOB SOURCE_FILES_IMAGE_HANDLING   ${SRC_DIR}/image_handling/*.cpp)
FILE(GLOB HEADER_FILES_IMAGE_HANDLING   ${SRC_DIR}/image_handling/*.h
                                        ${SRC_DIR}/image_handling/*.inl)

FILE(GLOB SOURCE_FILES_OBJ_HANDLING     ${SRC_DIR}/obj_handling/*.cpp)
FILE(GLOB HEADER_FILES_OBJ_HANDLING     ${SRC_DIR}/obj_handling/*.h
                                        ${SRC_DIR}/obj_handling/*.inl)

FILE(GLOB SHADER_FILES                  ${SRC_DIR}/shader/*.glsl)

# define source groups
SET(SOURCE_GROUP_DELIMITER "/")

SOURCE_GROUP(source_files                   FILES ${SOURCE_FILES}                   ${HEADER_FILES})
SOURCE_GROUP(source_files/image_handling    FILES ${SOURCE_FILES_IMAGE_HANDLING}    ${HEADER_FILES_IMAGE_HANDLING})
SOURCE_GROUP(source_files/obj_handling      FILES ${SOURCE_FILES_OBJ_HANDLING}      ${HEADER_FILES_OBJ_HANDLING})
SOURCE_GROUP(source_files/shader_files      FILES ${SHADER_FILES})

# combine source file lists
SET(SOURCE_FILES
    ${SOURCE_FILES}
    ${SOURCE_FILES_IMAGE_HANDLING}
    ${SOURCE_FILES_OBJ_HANDLING}
)

SET(HEADER_FILES
    ${HEADER_FILES}
    ${HEADER_FILES_IMAGE_HANDLING}
    ${HEADER_FILES_OBJ_HANDLING}
)

SET(SHADER_FILES
    ${SHADER_FILES}
)

# include header and inline files in source files for visual studio projects
IF (WIN32)
    IF (MSVC)
        SET (SOURCE_FILES ${SOURCE_FILES} ${HEADER_FILES} ${SHADER_FILES})
    ENDIF (MSVC)
ENDIF (WIN32)

# set include directories
INCLUDE_DIRECTORIES(
    ${GLOBAL_SRC_DIR}/app_geometry_correction
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
)
IF (WIN32)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}
        devIL
        glew32
        glut32
        optimized libboost_program_options-${SCHISM_BOOST_LIB_POSTFIX_REL}  debug libboost_program_options-${SCHISM_BOOST_LIB_POSTFIX_DBG}

    )
ENDIF (WIN32)
IF (UNIX)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}
        IL
        GLEW
        glut
        boost_program_options
    )
ENDIF (UNIX)

ADD_DEPENDENCIES(${PROJECT_NAME}
    scm_ogl
    scm_core
)
