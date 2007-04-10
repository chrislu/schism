
PROJECT(app_glut_tests)

SET(SRC_DIR "${GLOBAL_SRC_DIR}/app_glut_tests")

SET(EXECUTABLE_OUTPUT_PATH     ${SCHISM_EXECUTABLE_DIR})

# source files
FILE(GLOB SOURCE_FILES                  ${SRC_DIR}/ogl_08_fbo.cpp) #ogl_04_glsl_phong_shading.cpp)
FILE(GLOB HEADER_FILES                  ${SRC_DIR}/*.h)
FILE(GLOB INLINE_FILES                  ${SRC_DIR}/*.inl)

FILE(GLOB SOURCE_FILES_IMAGE_HANDLING   ${SRC_DIR}/image_handling/*.cpp)
FILE(GLOB HEADER_FILES_IMAGE_HANDLING   ${SRC_DIR}/image_handling/*.h)
FILE(GLOB INLINE_FILES_IMAGE_HANDLING   ${SRC_DIR}/image_handling/*.inl)

FILE(GLOB SOURCE_FILES_OBJ_HANDLING     ${SRC_DIR}/obj_handling/*.cpp)
FILE(GLOB HEADER_FILES_OBJ_HANDLING     ${SRC_DIR}/obj_handling/*.h)
FILE(GLOB INLINE_FILES_OBJ_HANDLING     ${SRC_DIR}/obj_handling/*.inl)

FILE(GLOB SHADER_FILES                  ${SRC_DIR}/shader/*.glsl)

# define source groups
SET(SOURCE_GROUP_DELIMITER "/")

SOURCE_GROUP(source_files                   FILES ${SOURCE_FILES}                   ${HEADER_FILES}                     ${INLINE_FILES})
SOURCE_GROUP(source_files/image_handling    FILES ${SOURCE_FILES_IMAGE_HANDLING}    ${HEADER_FILES_IMAGE_HANDLING}      ${INLINE_FILES_IMAGE_HANDLING})
SOURCE_GROUP(source_files/obj_handling      FILES ${SOURCE_FILES_OBJ_HANDLING}      ${HEADER_FILES_OBJ_HANDLING}        ${INLINE_FILES_OBJ_HANDLING})
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

SET(INLINE_FILES
    ${INLINE_FILES}
    ${INLINE_FILES_IMAGE_HANDLING}
    ${INLINE_FILES_OBJ_HANDLING}
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
    ${GLOBAL_SRC_DIR}/app_glut_tests
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
ADD_EXECUTABLE(app_glut_tests ${SOURCE_FILES})

# link libraries
TARGET_LINK_LIBRARIES(app_glut_tests
    lib_ogl
    lib_scm_core
)
IF (WIN32)
    TARGET_LINK_LIBRARIES(app_glut_tests
        devIL
        glew32
        glut32
    )
ENDIF (WIN32)
IF (UNIX)
    TARGET_LINK_LIBRARIES(app_glut_tests
        IL
        GLEW
        glut
    )
ENDIF (UNIX)

ADD_DEPENDENCIES(app_glut_tests
    lib_ogl
    lib_scm_core
)
