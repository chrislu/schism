PROJECT(lib_scm_core)

SET(SRC_DIR "${GLOBAL_SRC_DIR}/lib_scm_core")

SET(LIBRARY_OUTPUT_PATH     ${SCHISM_LIBRARY_DIR})

# source files
FILE(GLOB SOURCE_FILES                  ${SRC_DIR}/scm_core/*.cpp)
FILE(GLOB HEADER_FILES                  ${SRC_DIR}/scm_core/*.h)
FILE(GLOB INLINE_FILES                  ${SRC_DIR}/scm_core/*.inl)

FILE(GLOB SOURCE_FILES_CONSOLE          ${SRC_DIR}/scm_core/console/*.cpp)
FILE(GLOB HEADER_FILES_CONSOLE          ${SRC_DIR}/scm_core/console/*.h)
FILE(GLOB INLINE_FILES_CONSOLE          ${SRC_DIR}/scm_core/console/*.inl)
FILE(GLOB SOURCE_FILES_CONSOLE_DETAIL   ${SRC_DIR}/scm_core/console/detail/*.cpp)
FILE(GLOB HEADER_FILES_CONSOLE_DETAIL   ${SRC_DIR}/scm_core/console/detail/*.h)
FILE(GLOB INLINE_FILES_CONSOLE_DETAIL   ${SRC_DIR}/scm_core/console/detail/*.inl)

FILE(GLOB SOURCE_FILES_EXCEPTION        ${SRC_DIR}/scm_core/exception/*.cpp)
FILE(GLOB HEADER_FILES_EXCEPTION        ${SRC_DIR}/scm_core/exception/*.h)
FILE(GLOB INLINE_FILES_EXCEPTION        ${SRC_DIR}/scm_core/exception/*.inl)

FILE(GLOB SOURCE_FILES_MATH             ${SRC_DIR}/scm_core/math/*.cpp)
FILE(GLOB HEADER_FILES_MATH             ${SRC_DIR}/scm_core/math/*.h)
FILE(GLOB INLINE_FILES_MATH             ${SRC_DIR}/scm_core/math/*.inl)

FILE(GLOB SOURCE_FILES_PLATFORM                 ${SRC_DIR}/scm_core/platform/*.cpp)
FILE(GLOB HEADER_FILES_PLATFORM                 ${SRC_DIR}/scm_core/platform/*.h)
FILE(GLOB INLINE_FILES_PLATFORM                 ${SRC_DIR}/scm_core/platform/*.inl)
FILE(GLOB SOURCE_FILES_PLATFORM_GRAPHICS        ${SRC_DIR}/scm_core/platform/graphics/*.cpp)
FILE(GLOB HEADER_FILES_PLATFORM_GRAPHICS        ${SRC_DIR}/scm_core/platform/graphics/*.h)
FILE(GLOB INLINE_FILES_PLATFORM_GRAPHICS        ${SRC_DIR}/scm_core/platform/graphics/*.inl)
FILE(GLOB SOURCE_FILES_PLATFORM_GRAPHICS_DETAIL ${SRC_DIR}/scm_core/platform/graphics/detail/*.cpp)
FILE(GLOB HEADER_FILES_PLATFORM_GRAPHICS_DETAIL ${SRC_DIR}/scm_core/platform/graphics/detail/*.h)
FILE(GLOB INLINE_FILES_PLATFORM_GRAPHICS_DETAIL ${SRC_DIR}/scm_core/platform/graphics/detail/*.inl)

FILE(GLOB SOURCE_FILES_ROOT             ${SRC_DIR}/scm_core/root/*.cpp)
FILE(GLOB HEADER_FILES_ROOT             ${SRC_DIR}/scm_core/root/*.h)
FILE(GLOB INLINE_FILES_ROOT             ${SRC_DIR}/scm_core/root/*.inl)
FILE(GLOB SOURCE_FILES_ROOT_DETAIL      ${SRC_DIR}/scm_core/root/detail/*.cpp)
FILE(GLOB HEADER_FILES_ROOT_DETAIL      ${SRC_DIR}/scm_core/root/detail/*.h)
FILE(GLOB INLINE_FILES_ROOT_DETAIL      ${SRC_DIR}/scm_core/root/detail/*.inl)

FILE(GLOB SOURCE_FILES_SCRIPT           ${SRC_DIR}/scm_core/script/*.cpp)
FILE(GLOB HEADER_FILES_SCRIPT           ${SRC_DIR}/scm_core/script/*.h)
FILE(GLOB INLINE_FILES_SCRIPT           ${SRC_DIR}/scm_core/script/*.inl)
FILE(GLOB SOURCE_FILES_SCRIPT_DETAIL    ${SRC_DIR}/scm_core/script/detail/*.cpp)
FILE(GLOB HEADER_FILES_SCRIPT_DETAIL    ${SRC_DIR}/scm_core/script/detail/*.h)
FILE(GLOB INLINE_FILES_SCRIPT_DETAIL    ${SRC_DIR}/scm_core/script/detail/*.inl)

FILE(GLOB SOURCE_FILES_TIME             ${SRC_DIR}/scm_core/time/*.cpp)
FILE(GLOB HEADER_FILES_TIME             ${SRC_DIR}/scm_core/time/*.h)
FILE(GLOB INLINE_FILES_TIME             ${SRC_DIR}/scm_core/time/*.inl)
FILE(GLOB SOURCE_FILES_TIME_DETAIL      ${SRC_DIR}/scm_core/time/detail/*.cpp)
FILE(GLOB HEADER_FILES_TIME_DETAIL      ${SRC_DIR}/scm_core/time/detail/*.h)
FILE(GLOB INLINE_FILES_TIME_DETAIL      ${SRC_DIR}/scm_core/time/detail/*.inl)

FILE(GLOB SOURCE_FILES_UTILITIES        ${SRC_DIR}/scm_core/utilities/*.cpp)
FILE(GLOB HEADER_FILES_UTILITIES        ${SRC_DIR}/scm_core/utilities/*.h)
FILE(GLOB INLINE_FILES_UTILITIES        ${SRC_DIR}/scm_core/utilities/*.inl)


# define source groups
SET(SOURCE_GROUP_DELIMITER "/")

SOURCE_GROUP(source_files                           FILES ${SOURCE_FILES}                           ${HEADER_FILES}                             ${INLINE_FILES})
SOURCE_GROUP(source_files/console                   FILES ${SOURCE_FILES_CONSOLE}                   ${HEADER_FILES_CONSOLE}                     ${INLINE_FILES_CONSOLE})
SOURCE_GROUP(source_files/console/detail            FILES ${SOURCE_FILES_CONSOLE_DETAIL}            ${HEADER_FILES_CONSOLE_DETAIL}              ${INLINE_FILES_CONSOLE_DETAIL})
SOURCE_GROUP(source_files/exception                 FILES ${SOURCE_FILES_EXCEPTION}                 ${HEADER_FILES_EXCEPTION}                   ${INLINE_FILES_EXCEPTION})
SOURCE_GROUP(source_files/math                      FILES ${SOURCE_FILES_MATH}                      ${HEADER_FILES_MATH}                        ${INLINE_FILES_MATH})
SOURCE_GROUP(source_files/platform                  FILES ${SOURCE_FILES_PLATFORM}                  ${HEADER_FILES_PLATFORM}                    ${INLINE_FILES_PLATFORM})
SOURCE_GROUP(source_files/platform/graphics         FILES ${SOURCE_FILES_PLATFORM_GRAPHICS}         ${HEADER_FILES_PLATFORM_GRAPHICS}           ${INLINE_FILES_PLATFORM_GRAPHICS})
SOURCE_GROUP(source_files/platform/graphics/detail  FILES ${SOURCE_FILES_PLATFORM_GRAPHICS_DETAIL}  ${HEADER_FILES_PLATFORM_GRAPHICS_DETAIL}    ${INLINE_FILES_PLATFORM_GRAPHICS_DETAIL})
SOURCE_GROUP(source_files/root                      FILES ${SOURCE_FILES_ROOT}                      ${HEADER_FILES_ROOT}                        ${INLINE_FILES_ROOT})
SOURCE_GROUP(source_files/root/detail               FILES ${SOURCE_FILES_ROOT_DETAIL}               ${HEADER_FILES_ROOT_DETAIL}                 ${INLINE_FILES_ROOT_DETAIL})
SOURCE_GROUP(source_files/script                    FILES ${SOURCE_FILES_SCRIPT}                    ${HEADER_FILES_SCRIPT}                      ${INLINE_FILES_SCRIPT})
SOURCE_GROUP(source_files/script/detail             FILES ${SOURCE_FILES_SCRIPT_DETAIL}             ${HEADER_FILES_SCRIPT_DETAIL}               ${INLINE_FILES_SCRIPT_DETAIL})
SOURCE_GROUP(source_files/time                      FILES ${SOURCE_FILES_TIME}                      ${HEADER_FILES_TIME}                        ${INLINE_FILES_TIME})
SOURCE_GROUP(source_files/time/detail               FILES ${SOURCE_FILES_TIME_DETAIL}               ${HEADER_FILES_TIME_DETAIL}                 ${INLINE_FILES_TIME_DETAIL})
SOURCE_GROUP(source_files/utilities                 FILES ${SOURCE_FILES_UTILITIES}                 ${HEADER_FILES_UTILITIES}                   ${INLINE_FILES_UTILITIES})

# combine source file lists
SET(SOURCE_FILES
    ${SOURCE_FILES}
    ${SOURCE_FILES_EXCEPTION}
    ${SOURCE_FILES_MATH}
    ${SOURCE_FILES_PLATFORM}
    ${SOURCE_FILES_TIME}
    ${SOURCE_FILES_TIME_DETAIL}
    ${SOURCE_FILES_UTILITIES}
)

SET(HEADER_FILES
    ${HEADER_FILES}                
    ${HEADER_FILES_EXCEPTION}
    ${HEADER_FILES_MATH}
    ${HEADER_FILES_TIME}
    ${HEADER_FILES_TIME_DETAIL}
    ${HEADER_FILES_UTILITIES}
)

SET(INLINE_FILES
    ${INLINE_FILES}
    ${INLINE_FILES_EXCEPTION}
    ${INLINE_FILES_MATH}
    ${INLINE_FILES_TIME}
    ${INLINE_FILES_TIME_DETAIL}
    ${INLINE_FILES_UTILITIES}
)

IF (WIN32)
SET(SOURCE_FILES
    ${SOURCE_FILES}
    ${SOURCE_FILES_CONSOLE}
    ${SOURCE_FILES_CONSOLE_DETAIL}
    ${SOURCE_FILES_PLATFORM}
    ${SOURCE_FILES_PLATFORM_GRAPHICS}
    ${SOURCE_FILES_PLATFORM_GRAPHICS_DETAIL}
    ${SOURCE_FILES_ROOT}
    ${SOURCE_FILES_ROOT_DETAIL}
    ${SOURCE_FILES_SCRIPT}
    ${SOURCE_FILES_SCRIPT_DETAIL}
)

SET(HEADER_FILES
    ${HEADER_FILES}                
    ${HEADER_FILES_CONSOLE}
    ${HEADER_FILES_CONSOLE_DETAIL}
    ${HEADER_FILES_PLATFORM}
    ${HEADER_FILES_PLATFORM_GRAPHICS}
    ${HEADER_FILES_PLATFORM_GRAPHICS_DETAIL}
    ${HEADER_FILES_ROOT}
    ${HEADER_FILES_ROOT_DETAIL}
    ${HEADER_FILES_SCRIPT}
    ${HEADER_FILES_SCRIPT_DETAIL}
)
SET(INLINE_FILES
    ${INLINE_FILES}
    ${INLINE_FILES_CONSOLE}
    ${INLINE_FILES_CONSOLE_DETAIL}
    ${INLINE_FILES_PLATFORM}
    ${INLINE_FILES_PLATFORM_GRAPHICS}
    ${INLINE_FILES_PLATFORM_GRAPHICS_DETAIL}
    ${INLINE_FILES_ROOT}
    ${INLINE_FILES_ROOT_DETAIL}
    ${INLINE_FILES_SCRIPT}
    ${INLINE_FILES_SCRIPT_DETAIL}
)
ENDIF (WIN32)


# include header and inline files in source files for visual studio projects
IF (WIN32)
    IF (MSVC)
        SET (SOURCE_FILES ${SOURCE_FILES} ${HEADER_FILES} ${INLINE_FILES})
    ENDIF (MSVC)
ENDIF (WIN32)

# set include directories
INCLUDE_DIRECTORIES(
    ${GLOBAL_SRC_DIR}/lib_scm_core
)
# set include directories for windows external dependencies
IF (WIN32)
    INCLUDE_DIRECTORIES(
        ${GLOBAL_EXT_DIR}/inc
        ${GLOBAL_EXT_DIR}/inc/boost
        ${GLOBAL_EXT_DIR}/inc/lua
        ${GLOBAL_EXT_DIR}/inc/luabind
    )
ENDIF (WIN32)

# set library directories
LINK_DIRECTORIES(
)
# set library directories for windows external dependencies
IF (WIN32)
    LINK_DIRECTORIES(
        ${GLOBAL_EXT_DIR}/lib
    )
ENDIF (WIN32)

# add preprocessor defines
ADD_DEFINITIONS(-DSCM_CORE_BUILD)

# add/create library
ADD_LIBRARY(lib_scm_core SHARED ${SOURCE_FILES})

# link libraries
IF (WIN32)
    TARGET_LINK_LIBRARIES(lib_scm_core
        liblua
        libluabind
    )
ENDIF (WIN32)

IF (UNIX)
    TARGET_LINK_LIBRARIES(lib_scm_core
        #lua
    )
ENDIF (UNIX)
