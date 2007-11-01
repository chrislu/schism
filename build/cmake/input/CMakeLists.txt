
PROJECT(scm_input)

SET(SRC_DIR "${SCHISM_SRC_DIR}/scm/input")

SET(LIBRARY_OUTPUT_PATH                 ${SCHISM_LIBRARY_DIR})

# source files
FILE(GLOB SOURCE_FILES                  ${SRC_DIR}/*.cpp)
FILE(GLOB HEADER_FILES                  ${SRC_DIR}/*.h
                                        ${SRC_DIR}/*.inl)

FILE(GLOB SOURCE_FILES_TRACKING         ${SRC_DIR}/tracking/*.cpp)
FILE(GLOB HEADER_FILES_TRACKING         ${SRC_DIR}/tracking/*.h
                                        ${SRC_DIR}/tracking/*.inl)

FILE(GLOB SOURCE_FILES_TRACKING_DETAIL  ${SRC_DIR}/tracking/detail/*.cpp)
FILE(GLOB HEADER_FILES_TRACKING_DETAIL  ${SRC_DIR}/tracking/detail/*.h
                                        ${SRC_DIR}/tracking/detail*.inl)



# define source grous
SET(SOURCE_GROUP_DELIMITER "/")

SOURCE_GROUP(source_files						FILES ${SOURCE_FILES}					${HEADER_FILES})
SOURCE_GROUP(source_files/tracking              FILES ${SOURCE_FILES_TRACKING}          ${HEADER_FILES_TRACKING})
SOURCE_GROUP(source_files/tracking/detail       FILES ${SOURCE_FILES_TRACKING_DETAIL}   ${HEADER_FILES_TRACKING_DETAIL})

# combine source file lists
SET(SOURCE_FILES
    ${SOURCE_FILES}
    ${SOURCE_FILES_TRACKING}
    ${SOURCE_FILES_TRACKING_DETAIL}
)

SET(HEADER_FILES
    ${HEADER_FILES}
    ${HEADER_FILES_TRACKING}
    ${HEADER_FILES_TRACKING_DETAIL}
)

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
ENDIF (WIN32)

IF (WIN32)
    # add preprocessor defines
    ADD_DEFINITIONS(-DSCM_BUILD_LIBRARY_input=\(\))
ENDIF (WIN32)

# add/create library
ADD_LIBRARY(${PROJECT_NAME} SHARED ${SOURCE_FILES})

# link libraries
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
    scm_core
)
IF (WIN32)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}
        ws2_32
    )
ENDIF (WIN32)
IF (UNIX)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}
    )
ENDIF (UNIX)

ADD_DEPENDENCIES(${PROJECT_NAME}
    scm_core
)
