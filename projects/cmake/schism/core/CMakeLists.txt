PROJECT(scm_core)

SET(SRC_DIR "${GLOBAL_SRC_DIR}/schism/scm/core")

SET(LIBRARY_OUTPUT_PATH                         ${SCHISM_LIBRARY_DIR})

# source files
FILE(GLOB SOURCE_FILES_SCM                      ${GLOBAL_SRC_DIR}/schism/scm/*.cpp)
FILE(GLOB HEADER_FILES_SCM                      ${GLOBAL_SRC_DIR}/schism/scm/*.h)
FILE(GLOB INLINE_FILES_SCM                      ${GLOBAL_SRC_DIR}/schism/scm/*.inl)

FILE(GLOB SOURCE_FILES                          ${SRC_DIR}/*.cpp)
FILE(GLOB HEADER_FILES                          ${SRC_DIR}/*.h)
FILE(GLOB INLINE_FILES                          ${SRC_DIR}/*.inl)

FILE(GLOB SOURCE_FILES_CONSOLE                  ${SRC_DIR}/console/*.cpp)
FILE(GLOB HEADER_FILES_CONSOLE                  ${SRC_DIR}/console/*.h)
FILE(GLOB INLINE_FILES_CONSOLE                  ${SRC_DIR}/console/*.inl)
FILE(GLOB SOURCE_FILES_CONSOLE_DETAIL           ${SRC_DIR}/console/detail/*.cpp)
FILE(GLOB HEADER_FILES_CONSOLE_DETAIL           ${SRC_DIR}/console/detail/*.h)
FILE(GLOB INLINE_FILES_CONSOLE_DETAIL           ${SRC_DIR}/console/detail/*.inl)

FILE(GLOB SOURCE_FILES_ROOT                     ${SRC_DIR}/root/*.cpp)
FILE(GLOB HEADER_FILES_ROOT                     ${SRC_DIR}/root/*.h)
FILE(GLOB INLINE_FILES_ROOT                     ${SRC_DIR}/root/*.inl)
FILE(GLOB SOURCE_FILES_ROOT_DETAIL              ${SRC_DIR}/root/detail/*.cpp)
FILE(GLOB HEADER_FILES_ROOT_DETAIL              ${SRC_DIR}/root/detail/*.h)
FILE(GLOB INLINE_FILES_ROOT_DETAIL              ${SRC_DIR}/root/detail/*.inl)

FILE(GLOB SOURCE_FILES_EXCEPTION                ${SRC_DIR}/exception/*.cpp)
FILE(GLOB HEADER_FILES_EXCEPTION                ${SRC_DIR}/exception/*.h)
FILE(GLOB INLINE_FILES_EXCEPTION                ${SRC_DIR}/exception/*.inl)

FILE(GLOB SOURCE_FILES_FONT                     ${SRC_DIR}/font/*.cpp)
FILE(GLOB HEADER_FILES_FONT                     ${SRC_DIR}/font/*.h)
FILE(GLOB INLINE_FILES_FONT                     ${SRC_DIR}/font/*.inl)
FILE(GLOB SOURCE_FILES_FONT_DETAIL              ${SRC_DIR}/font/detail/*.cpp)
FILE(GLOB HEADER_FILES_FONT_DETAIL              ${SRC_DIR}/font/detail/*.h)
FILE(GLOB INLINE_FILES_FONT_DETAIL              ${SRC_DIR}/font/detail/*.inl)

FILE(GLOB SOURCE_FILES_GUI						${SRC_DIR}/gui/*.cpp)
FILE(GLOB HEADER_FILES_GUI						${SRC_DIR}/gui/*.h)
FILE(GLOB INLINE_FILES_GUI						${SRC_DIR}/gui/*.inl)
FILE(GLOB SOURCE_FILES_GUI_DETAIL				${SRC_DIR}/gui/detail/*.cpp)
FILE(GLOB HEADER_FILES_GUI_DETAIL				${SRC_DIR}/gui/detail/*.h)
FILE(GLOB INLINE_FILES_GUI_DETAIL				${SRC_DIR}/gui/detail/*.inl)

FILE(GLOB SOURCE_FILES_MATH                     ${SRC_DIR}/math/*.cpp)
FILE(GLOB HEADER_FILES_MATH                     ${SRC_DIR}/math/*.h)
FILE(GLOB INLINE_FILES_MATH                     ${SRC_DIR}/math/*.inl)

FILE(GLOB SOURCE_FILES_PLATFORM                 ${SRC_DIR}/platform/*.cpp)
FILE(GLOB HEADER_FILES_PLATFORM                 ${SRC_DIR}/platform/*.h)
FILE(GLOB INLINE_FILES_PLATFORM                 ${SRC_DIR}/platform/*.inl)
FILE(GLOB SOURCE_FILES_PLATFORM_GRAPHICS        ${SRC_DIR}/platform/graphics/*.cpp)
FILE(GLOB HEADER_FILES_PLATFORM_GRAPHICS        ${SRC_DIR}/platform/graphics/*.h)
FILE(GLOB INLINE_FILES_PLATFORM_GRAPHICS        ${SRC_DIR}/platform/graphics/*.inl)
FILE(GLOB SOURCE_FILES_PLATFORM_GRAPHICS_DETAIL ${SRC_DIR}/platform/graphics/detail/*.cpp)
FILE(GLOB HEADER_FILES_PLATFORM_GRAPHICS_DETAIL ${SRC_DIR}/platform/graphics/detail/*.h)
FILE(GLOB INLINE_FILES_PLATFORM_GRAPHICS_DETAIL ${SRC_DIR}/platform/graphics/detail/*.inl)

FILE(GLOB SOURCE_FILES_RESOURCE                 ${SRC_DIR}/resource/*.cpp)
FILE(GLOB HEADER_FILES_RESOURCE                 ${SRC_DIR}/resource/*.h)
FILE(GLOB INLINE_FILES_RESOURCE                 ${SRC_DIR}/resource/*.inl)
FILE(GLOB SOURCE_FILES_RESOURCE_DETAIL          ${SRC_DIR}/resource/detail/*.cpp)
FILE(GLOB HEADER_FILES_RESOURCE_DETAIL          ${SRC_DIR}/resource/detail/*.h)
FILE(GLOB INLINE_FILES_RESOURCE_DETAIL          ${SRC_DIR}/resource/detail/*.inl)

FILE(GLOB SOURCE_FILES_SCRIPT                   ${SRC_DIR}/script/*.cpp)
FILE(GLOB HEADER_FILES_SCRIPT                   ${SRC_DIR}/script/*.h)
FILE(GLOB INLINE_FILES_SCRIPT                   ${SRC_DIR}/script/*.inl)
FILE(GLOB SOURCE_FILES_SCRIPT_DETAIL            ${SRC_DIR}/script/detail/*.cpp)
FILE(GLOB HEADER_FILES_SCRIPT_DETAIL            ${SRC_DIR}/script/detail/*.h)
FILE(GLOB INLINE_FILES_SCRIPT_DETAIL            ${SRC_DIR}/script/detail/*.inl)

FILE(GLOB SOURCE_FILES_TIME                     ${SRC_DIR}/time/*.cpp)
FILE(GLOB HEADER_FILES_TIME                     ${SRC_DIR}/time/*.h)
FILE(GLOB INLINE_FILES_TIME                     ${SRC_DIR}/time/*.inl)
FILE(GLOB SOURCE_FILES_TIME_DETAIL              ${SRC_DIR}/time/detail/*.cpp)
FILE(GLOB HEADER_FILES_TIME_DETAIL              ${SRC_DIR}/time/detail/*.h)
FILE(GLOB INLINE_FILES_TIME_DETAIL              ${SRC_DIR}/time/detail/*.inl)

FILE(GLOB SOURCE_FILES_UTILITIES                ${SRC_DIR}/utilities/*.cpp)
FILE(GLOB HEADER_FILES_UTILITIES                ${SRC_DIR}/utilities/*.h)
FILE(GLOB INLINE_FILES_UTILITIES                ${SRC_DIR}/utilities/*.inl)


# define source groups
SET(SOURCE_GROUP_DELIMITER "/")

SOURCE_GROUP(scm                                    FILES ${SOURCE_FILES_SCM}                       ${HEADER_FILES_SCM}                         ${INLINE_FILES_SCM})
SOURCE_GROUP(source_files                           FILES ${SOURCE_FILES}                           ${HEADER_FILES}                             ${INLINE_FILES})
SOURCE_GROUP(source_files/console                   FILES ${SOURCE_FILES_CONSOLE}                   ${HEADER_FILES_CONSOLE}                     ${INLINE_FILES_CONSOLE})
SOURCE_GROUP(source_files/console/detail            FILES ${SOURCE_FILES_CONSOLE_DETAIL}            ${HEADER_FILES_CONSOLE_DETAIL}              ${INLINE_FILES_CONSOLE_DETAIL})
SOURCE_GROUP(source_files/exception                 FILES ${SOURCE_FILES_EXCEPTION}                 ${HEADER_FILES_EXCEPTION}                   ${INLINE_FILES_EXCEPTION})
SOURCE_GROUP(source_files/font                      FILES ${SOURCE_FILES_FONT}                      ${HEADER_FILES_FONT}                        ${INLINE_FILES_FONT})
SOURCE_GROUP(source_files/font/detail               FILES ${SOURCE_FILES_FONT_DETAIL}               ${HEADER_FILES_FONT_DETAIL}                 ${INLINE_FILES_FONT_DETAIL})
SOURCE_GROUP(source_files/gui						FILES ${SOURCE_FILES_GUI}                       ${HEADER_FILES_GUI}                         ${INLINE_FILES_GUI})
SOURCE_GROUP(source_files/gui/detail				FILES ${SOURCE_FILES_GUI_DETAIL}                ${HEADER_FILES_GUI_DETAIL}                  ${INLINE_FILES_GUI_DETAIL})
SOURCE_GROUP(source_files/math                      FILES ${SOURCE_FILES_MATH}                      ${HEADER_FILES_MATH}                        ${INLINE_FILES_MATH})
SOURCE_GROUP(source_files/platform                  FILES ${SOURCE_FILES_PLATFORM}                  ${HEADER_FILES_PLATFORM}                    ${INLINE_FILES_PLATFORM})
SOURCE_GROUP(source_files/platform/graphics         FILES ${SOURCE_FILES_PLATFORM_GRAPHICS}         ${HEADER_FILES_PLATFORM_GRAPHICS}           ${INLINE_FILES_PLATFORM_GRAPHICS})
SOURCE_GROUP(source_files/platform/graphics/detail  FILES ${SOURCE_FILES_PLATFORM_GRAPHICS_DETAIL}  ${HEADER_FILES_PLATFORM_GRAPHICS_DETAIL}    ${INLINE_FILES_PLATFORM_GRAPHICS_DETAIL})
SOURCE_GROUP(source_files/resource                  FILES ${SOURCE_FILES_RESOURCE}                  ${HEADER_FILES_RESOURCE}                    ${INLINE_FILES_RESOURCE})
SOURCE_GROUP(source_files/resource/detail           FILES ${SOURCE_FILES_RESOURCE_DETAIL}           ${HEADER_FILES_RESOURCE_DETAIL}             ${INLINE_FILES_RESOURCE_DETAIL})
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
    ${SOURCE_FILES_SCM}
    ${SOURCE_FILES_CONSOLE}
    ${SOURCE_FILES_CONSOLE_DETAIL}
    ${SOURCE_FILES_EXCEPTION}
    ${SOURCE_FILES_MATH}
    ${SOURCE_FILES_RESOURCE}
    ${SOURCE_FILES_RESOURCE_DETAIL}
    ${SOURCE_FILES_ROOT}
    ${SOURCE_FILES_ROOT_DETAIL}
    ${SOURCE_FILES_PLATFORM}
    ${SOURCE_FILES_PLATFORM_GRAPHICS}
    ${SOURCE_FILES_PLATFORM_GRAPHICS_DETAIL}
    ${SOURCE_FILES_TIME}
    ${SOURCE_FILES_TIME_DETAIL}
    ${SOURCE_FILES_UTILITIES}
)

SET(HEADER_FILES
    ${HEADER_FILES}
    ${HEADER_FILES_SCM}
    ${HEADER_FILES_CONSOLE}
    ${HEADER_FILES_CONSOLE_DETAIL}
    ${HEADER_FILES_EXCEPTION}
    ${HEADER_FILES_MATH}
    ${HEADER_FILES_RESOURCE}
    ${HEADER_FILES_RESOURCE_DETAIL}
    ${HEADER_FILES_ROOT}
    ${HEADER_FILES_ROOT_DETAIL}
    ${HEADER_FILES_PLATFORM}
    ${HEADER_FILES_PLATFORM_GRAPHICS}
    ${HEADER_FILES_PLATFORM_GRAPHICS_DETAIL}
    ${HEADER_FILES_TIME}
    ${HEADER_FILES_TIME_DETAIL}
    ${HEADER_FILES_UTILITIES}
)

SET(INLINE_FILES
    ${INLINE_FILES}
    ${INLINE_FILES_SCM}
    ${INLINE_FILES_CONSOLE}
    ${INLINE_FILES_CONSOLE_DETAIL}
    ${INLINE_FILES_EXCEPTION}
    ${INLINE_FILES_MATH}
    ${INLINE_FILES_RESOURCE}
    ${INLINE_FILES_RESOURCE_DETAIL}
    ${INLINE_FILES_ROOT}
    ${INLINE_FILES_ROOT_DETAIL}
    ${INLINE_FILES_PLATFORM}
    ${INLINE_FILES_PLATFORM_GRAPHICS}
    ${INLINE_FILES_PLATFORM_GRAPHICS_DETAIL}
    ${INLINE_FILES_TIME}
    ${INLINE_FILES_TIME_DETAIL}
    ${INLINE_FILES_UTILITIES}
)

IF (WIN32)
SET(SOURCE_FILES
    ${SOURCE_FILES}
    ${SOURCE_FILES_FONT}
    ${SOURCE_FILES_FONT_DETAIL}
    ${SOURCE_FILES_GUI}
    ${SOURCE_FILES_GUI_DETAIL}
    ${SOURCE_FILES_SCRIPT}
    ${SOURCE_FILES_SCRIPT_DETAIL}
)

SET(HEADER_FILES
    ${HEADER_FILES}                
    ${HEADER_FILES_FONT}
    ${HEADER_FILES_FONT_DETAIL}
    ${HEADER_FILES_GUI}
    ${HEADER_FILES_GUI_DETAIL}
    ${HEADER_FILES_SCRIPT}
    ${HEADER_FILES_SCRIPT_DETAIL}
)
SET(INLINE_FILES
    ${INLINE_FILES}
    ${INLINE_FILES_FONT}
    ${INLINE_FILES_FONT_DETAIL}
    ${INLINE_FILES_GUI}
    ${INLINE_FILES_GUI_DETAIL}
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
    ${GLOBAL_SRC_DIR}/schism
)
# set include directories for windows external dependencies
IF (WIN32)
    INCLUDE_DIRECTORIES(
        ${GLOBAL_EXT_DIR}/inc
        ${GLOBAL_EXT_DIR}/inc/boost
        ${GLOBAL_EXT_DIR}/inc/freetype
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

IF (WIN32)
    # add preprocessor defines
    ADD_DEFINITIONS(-DSCM_BUILD_LIBRARY_core=\(\))
ENDIF (WIN32)

# add/create library
ADD_LIBRARY(${PROJECT_NAME} SHARED ${SOURCE_FILES})

# link libraries
IF (WIN32)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}
        freetype2
        liblua
        libluabind
        optimized libboost_signals-${SCHISM_BOOST_LIB_POSTFIX_REL}      debug libboost_signals-${SCHISM_BOOST_LIB_POSTFIX_DBG}
        optimized libboost_date_time-${SCHISM_BOOST_LIB_POSTFIX_REL}    debug libboost_date_time-${SCHISM_BOOST_LIB_POSTFIX_DBG}
        optimized libboost_filesystem-${SCHISM_BOOST_LIB_POSTFIX_REL}   debug libboost_filesystem-${SCHISM_BOOST_LIB_POSTFIX_DBG}
    )
ENDIF (WIN32)

IF (UNIX)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}
        boost_date_time
        lua
        rt
        boost_signals
        boost_date_time
        boost_filesystem
    )
ENDIF (UNIX)
