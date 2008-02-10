
PROJECT(scm_ogl)

SET(SRC_DIR "${SCHISM_SRC_DIR}/scm/ogl")

SET(LIBRARY_OUTPUT_PATH                 ${SCHISM_LIBRARY_DIR})

# source files
FILE(GLOB SOURCE_FILES                  ${SRC_DIR}/*.cpp)
FILE(GLOB HEADER_FILES                  ${SRC_DIR}/*.h
                                        ${SRC_DIR}/*.inl)

FILE(GLOB SOURCE_FILES_BUFFER_OBJECT	${SRC_DIR}/buffer_object/*.cpp)
FILE(GLOB HEADER_FILES_BUFFER_OBJECT	${SRC_DIR}/buffer_object/*.h)

FILE(GLOB SOURCE_FILES_FONT             ${SRC_DIR}/font/*.cpp)
FILE(GLOB HEADER_FILES_FONT             ${SRC_DIR}/font/*.h)

FILE(GLOB SOURCE_FILES_FB_OBJECT        ${SRC_DIR}/framebuffer_object/*.cpp)
FILE(GLOB HEADER_FILES_FB_OBJECT        ${SRC_DIR}/framebuffer_object/*.h)

FILE(GLOB SOURCE_FILES_GUI				${SRC_DIR}/gui/*.cpp)
FILE(GLOB HEADER_FILES_GUI				${SRC_DIR}/gui/*.h)

FILE(GLOB SOURCE_FILES_MANIPULATORS     ${SRC_DIR}/manipulators/*.cpp)
FILE(GLOB HEADER_FILES_MANIPULATORS     ${SRC_DIR}/manipulators/*.h)

FILE(GLOB SOURCE_FILES_SHADER_OBJECTS   ${SRC_DIR}/shader_objects/*.cpp)
FILE(GLOB HEADER_FILES_SHADER_OBJECTS   ${SRC_DIR}/shader_objects/*.h)

FILE(GLOB SOURCE_FILES_SYSTEM			${SRC_DIR}/system/*.cpp)
FILE(GLOB HEADER_FILES_SYSTEM			${SRC_DIR}/system/*.h)

FILE(GLOB SOURCE_FILES_TEXTURES         ${SRC_DIR}/textures/*.cpp)
FILE(GLOB HEADER_FILES_TEXTURES         ${SRC_DIR}/textures/*.h)

FILE(GLOB SOURCE_FILES_TIME             ${SRC_DIR}/time/*.cpp)
FILE(GLOB HEADER_FILES_TIME             ${SRC_DIR}/time/*.h)

FILE(GLOB SOURCE_FILES_UTILITIES        ${SRC_DIR}/utilities/*.cpp)
FILE(GLOB HEADER_FILES_UTILITIES        ${SRC_DIR}/utilities/*.h)

FILE(GLOB SOURCE_FILES_VB_OBJECT        ${SRC_DIR}/vertexbuffer_object/*.cpp)
FILE(GLOB HEADER_FILES_VB_OBJECT        ${SRC_DIR}/vertexbuffer_object/*.h)

FILE(GLOB HEADER_FILES_MATH             ${SRC_DIR}/math/*.h
                                        ${SRC_DIR}/math/*.inl)

# define source grous
SET(SOURCE_GROUP_DELIMITER "/")

SOURCE_GROUP(source_files						FILES ${SOURCE_FILES}					${HEADER_FILES})
SOURCE_GROUP(source_files/font					FILES ${SOURCE_FILES_FONT}              ${HEADER_FILES_FONT})
SOURCE_GROUP(source_files/buffer_object			FILES ${SOURCE_FILES_BUFFER_OBJECT}		${HEADER_FILES_BUFFER_OBJECT})
SOURCE_GROUP(source_files/framebuffer_object	FILES ${SOURCE_FILES_FB_OBJECT}         ${HEADER_FILES_FB_OBJECT})
SOURCE_GROUP(source_files/gui					FILES ${SOURCE_FILES_GUI}				${HEADER_FILES_GUI})
SOURCE_GROUP(source_files/manipulators			FILES ${SOURCE_FILES_MANIPULATORS}      ${HEADER_FILES_MANIPULATORS})
SOURCE_GROUP(source_files/math					FILES                                   ${HEADER_FILES_MATH})
SOURCE_GROUP(source_files/shader_objects		FILES ${SOURCE_FILES_SHADER_OBJECTS}    ${HEADER_FILES_SHADER_OBJECTS})
SOURCE_GROUP(source_files/system				FILES ${SOURCE_FILES_SYSTEM}			${HEADER_FILES_SYSTEM})
SOURCE_GROUP(source_files/textures				FILES ${SOURCE_FILES_TEXTURES}          ${HEADER_FILES_TEXTURES})
SOURCE_GROUP(source_files/time					FILES ${SOURCE_FILES_TIME}              ${HEADER_FILES_TIME})
SOURCE_GROUP(source_files/vertexbuffer_object	FILES ${SOURCE_FILES_VB_OBJECT}         ${HEADER_FILES_VB_OBJECT})
SOURCE_GROUP(source_files/utilities				FILES ${SOURCE_FILES_UTILITIES}         ${HEADER_FILES_UTILITIES})

# combine source file lists
SET(SOURCE_FILES
    ${SOURCE_FILES}
    ${SOURCE_FILES_BUFFER_OBJECT}
    ${SOURCE_FILES_FB_OBJECT}
    ${SOURCE_FILES_MANIPULATORS}
    ${SOURCE_FILES_SHADER_OBJECTS}
    ${SOURCE_FILES_SYSTEM}
    ${SOURCE_FILES_TEXTURES}
    ${SOURCE_FILES_TIME}
    ${SOURCE_FILES_VB_OBJECT}
    ${SOURCE_FILES_UTILITIES}
)

SET(HEADER_FILES
    ${HEADER_FILES}
    ${HEADER_FILES_BUFFER_OBJECT}
    ${HEADER_FILES_FB_OBJECT}
    ${HEADER_FILES_MANIPULATORS}
    ${HEADER_FILES_MATH}
    ${HEADER_FILES_SHADER_OBJECTS}
    ${HEADER_FILES_SYSTEM}
    ${HEADER_FILES_TEXTURES}
    ${HEADER_FILES_TIME}
    ${HEADER_FILES_VB_OBJECT}
    ${HEADER_FILES_UTILITIES}
)

#IF (WIN32)
SET(SOURCE_FILES
    ${SOURCE_FILES}
    ${SOURCE_FILES_FONT}
    ${SOURCE_FILES_GUI}
)
SET(HEADER_FILES
    ${HEADER_FILES}
    ${HEADER_FILES_FONT}
    ${HEADER_FILES_GUI}
)
#ENDIF (WIN32)

# include header and inline files in source files for visual studio projects
IF (WIN32)
    IF (MSVC)
        SET (SOURCE_FILES ${SOURCE_FILES} ${HEADER_FILES})
    ENDIF (MSVC)
ENDIF (WIN32)

# set include and lib directories
INCLUDE_DIRECTORIES(
    ${SCHISM_SRC_DIR}
)
# set include directories for windows external dependencies
IF (WIN32)
    INCLUDE_DIRECTORIES(
        ${GLOBAL_EXT_DIR}/inc
        ${GLOBAL_EXT_DIR}/inc/boost
    )
ELSEIF (UNIX)
    INCLUDE_DIRECTORIES(
        /opt/boost/latest/inc
    )
ENDIF (WIN32)

# set library directories
LINK_DIRECTORIES(
    ${SCHISM_LIB_DIR}/${SCHISM_PLATFORM}
)
# set library directories for windows external dependencies
IF (WIN32)
    LINK_DIRECTORIES(
        ${GLOBAL_EXT_DIR}/lib
    )
ELSEIF (UNIX)
    LINK_DIRECTORIES(
        /opt/boost/latest/lib
    )
ENDIF (WIN32)

IF (WIN32)
    # add preprocessor defines
    ADD_DEFINITIONS(-DSCM_BUILD_LIBRARY_ogl=\(\))
ENDIF (WIN32)

# add/create library
ADD_LIBRARY(${PROJECT_NAME} SHARED ${SOURCE_FILES})

# link libraries
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
    scm_core
)
IF (WIN32)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}
        glew32
        glut32
        opengl32
    )
ENDIF (WIN32)
IF (UNIX)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}
        GLEW
        glut
    )
ENDIF (UNIX)

ADD_DEPENDENCIES(${PROJECT_NAME}
    scm_core
)
