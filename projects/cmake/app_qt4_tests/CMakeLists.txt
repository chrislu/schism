
PROJECT(app_qt4_tests)

SET(SRC_DIR "${GLOBAL_SRC_DIR}/app_qt4_tests")

SET(EXECUTABLE_OUTPUT_PATH     ${SCHISM_EXECUTABLE_DIR})

# source files
FILE(GLOB SOURCE_FILES                  ${SRC_DIR}/*.cpp)
FILE(GLOB HEADER_FILES                  ${SRC_DIR}/*.h)
FILE(GLOB INLINE_FILES                  ${SRC_DIR}/*.inl)

# define source groups
SET(SOURCE_GROUP_DELIMITER "/")

SOURCE_GROUP(source_files                   FILES ${SOURCE_FILES}                   ${HEADER_FILES}                     ${INLINE_FILES})

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

# include header and inline files in source files for visual studio projects
IF (WIN32)
    IF (MSVC)
        SET (SOURCE_FILES ${SOURCE_FILES} ${HEADER_FILES} ${INLINE_FILES} ${SHADER_FILES})
    ENDIF (MSVC)
ENDIF (WIN32)

# set include directories
INCLUDE_DIRECTORIES(
    ${GLOBAL_SRC_DIR}/app_qtt4_tests
    ${GLOBAL_SRC_DIR}/schism
)
# set include directories for windows external dependencies
IF (WIN32)
    INCLUDE_DIRECTORIES(
        ${GLOBAL_EXT_DIR}/inc
        ${GLOBAL_EXT_DIR}/inc/boost
        ${GLOBAL_EXT_DIR}/inc/qt
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
    optimized qtmain        debug qtmaind
    optimized QtCore4       debug QtCored4
    optimized QtGui4        debug QtGuid4
)

IF (WIN32)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}
        optimized boost_thread-${SCHISM_BOOST_LIB_POSTFIX_REL}					debug boost_thread-${SCHISM_BOOST_LIB_POSTFIX_DBG}
        optimized libboost_program_options-${SCHISM_BOOST_LIB_POSTFIX_REL}      debug libboost_program_options-${SCHISM_BOOST_LIB_POSTFIX_DBG}
    )
ENDIF (WIN32)

IF (UNIX)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}
        boost_program_options
    )
ENDIF (UNIX)

ADD_DEPENDENCIES(${PROJECT_NAME}
    scm_core
)
