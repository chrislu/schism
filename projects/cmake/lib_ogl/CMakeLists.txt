
PROJECT(lib_ogl)

SET(SRC_DIR "${GLOBAL_SRC_DIR}/lib_ogl")

SET(LIBRARY_OUTPUT_PATH     ${SCHISM_LIBRARY_DIR})

# source files
FILE(GLOB SOURCE_FILES                  ${SRC_DIR}/ogl/*.cpp)
FILE(GLOB HEADER_FILES                  ${SRC_DIR}/ogl/*.h)
FILE(GLOB INLINE_FILES                  ${SRC_DIR}/ogl/*.inl)

FILE(GLOB SOURCE_FILES_FB_OBJECT        ${SRC_DIR}/ogl/framebuffer_object/*.cpp)
FILE(GLOB HEADER_FILES_FB_OBJECT        ${SRC_DIR}/ogl/framebuffer_object/*.h)

FILE(GLOB SOURCE_FILES_MANIPULATORS     ${SRC_DIR}/ogl/manipulators/*.cpp)
FILE(GLOB HEADER_FILES_MANIPULATORS     ${SRC_DIR}/ogl/manipulators/*.h)

FILE(GLOB SOURCE_FILES_SHADER_OBJECTS   ${SRC_DIR}/ogl/shader_objects/*.cpp)
FILE(GLOB HEADER_FILES_SHADER_OBJECTS   ${SRC_DIR}/ogl/shader_objects/*.h)

FILE(GLOB SOURCE_FILES_TEXTURES         ${SRC_DIR}/ogl/textures/*.cpp)
FILE(GLOB HEADER_FILES_TEXTURES         ${SRC_DIR}/ogl/textures/*.h)

FILE(GLOB SOURCE_FILES_TIME             ${SRC_DIR}/ogl/time/*.cpp)
FILE(GLOB HEADER_FILES_TIME             ${SRC_DIR}/ogl/time/*.h)

FILE(GLOB SOURCE_FILES_UTILITIES        ${SRC_DIR}/ogl/utilities/*.cpp)
FILE(GLOB HEADER_FILES_UTILITIES        ${SRC_DIR}/ogl/utilities/*.h)

FILE(GLOB HEADER_FILES_MATH             ${SRC_DIR}/ogl/math/*.h)
FILE(GLOB INLINE_FILES_MATH             ${SRC_DIR}/ogl/math/*.inl)

# define source grous
SET(SOURCE_GROUP_DELIMITER "/")

SOURCE_GROUP(source_files                   FILES ${SOURCE_FILES}                   ${HEADER_FILES}                     ${INLINE_FILES})
SOURCE_GROUP(source_files/framebuffer_object FILES ${SOURCE_FILES_FB_OBJECT}        ${HEADER_FILES_FB_OBJECT})
SOURCE_GROUP(source_files/manipulators      FILES ${SOURCE_FILES_MANIPULATORS}      ${HEADER_FILES_MANIPULATORS})
SOURCE_GROUP(source_files/math              FILES                                   ${HEADER_FILES_MATH}                ${INLINE_FILES_MATH})
SOURCE_GROUP(source_files/shader_objects    FILES ${SOURCE_FILES_SHADER_OBJECTS}    ${HEADER_FILES_SHADER_OBJECTS})
SOURCE_GROUP(source_files/textures          FILES ${SOURCE_FILES_TEXTURES}          ${HEADER_FILES_TEXTURES})
SOURCE_GROUP(source_files/time              FILES ${SOURCE_FILES_TIME}              ${HEADER_FILES_TIME})
SOURCE_GROUP(source_files/utilities         FILES ${SOURCE_FILES_UTILITIES}         ${HEADER_FILES_UTILITIES})

# combine source file lists
SET(SOURCE_FILES
    ${SOURCE_FILES}
    ${SOURCE_FILES_FB_OBJECT}
    ${SOURCE_FILES_MANIPULATORS}
    ${SOURCE_FILES_SHADER_OBJECTS}
    ${SOURCE_FILES_TEXTURES}
    ${SOURCE_FILES_TIME}
    ${SOURCE_FILES_UTILITIES}
)

SET(HEADER_FILES
    ${HEADER_FILES}
    ${HEADER_FILES_FB_OBJECT}
    ${HEADER_FILES_MANIPULATORS}
    ${HEADER_FILES_MATH}
    ${HEADER_FILES_SHADER_OBJECTS}
    ${HEADER_FILES_TEXTURES}
    ${HEADER_FILES_TIME}
    ${HEADER_FILES_UTILITIES}
)

SET(INLINE_FILES
    ${INLINE_FILES}
    ${INLINE_FILES_MATH}
)

# include header and inline files in source files for visual studio projects
IF (WIN32)
    IF (MSVC)
        SET (SOURCE_FILES ${SOURCE_FILES} ${HEADER_FILES} ${INLINE_FILES})
    ENDIF (MSVC)
ENDIF (WIN32)

# set include and lib directories
INCLUDE_DIRECTORIES(
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

# add/create library
ADD_LIBRARY(lib_ogl STATIC ${SOURCE_FILES})

ADD_DEPENDENCIES(lib_ogl
    lib_scm_core
)
