#Project Name
PROJECT(schism)

# include platform definitions
INCLUDE(schism_platform)
INCLUDE(schism)

#-----------------------------------------------------------------------------
# Setup schism_VR version strings
SET(SCHISM_MAJOR_VERSION 0)
SET(SCHISM_MINOR_VERSION 0)
SET(SCHISM_BUILD_VERSION 1)
SET(SCHISM_VERSION "${schism_MAJOR_VERSION}.${schism_MINOR_VERSION}.${schism_BUILD_VERSION}")

#-----------------------------------------------------------------------------
# setup output directories.
IF(NOT LIBRARY_OUTPUT_PATH)
  SET(LIBRARY_OUTPUT_PATH ${schism_SOURCE_DIR}/../../lib/${SCHISM_PLATFORM} CACHE INTERNAL "Single output directory for building all libraries.")
ENDIF(NOT LIBRARY_OUTPUT_PATH)

IF(NOT EXECUTABLE_OUTPUT_PATH)
  SET(EXECUTABLE_OUTPUT_PATH ${schism_SOURCE_DIR}/../../bin/${SCHISM_PLATFORM} CACHE INTERNAL "Single output directory for building all executables.")
ENDIF(NOT EXECUTABLE_OUTPUT_PATH)

SET(SCHISM_LIBRARY_DIR    ${LIBRARY_OUTPUT_PATH})
SET(SCHISM_EXECUTABLE_DIR ${EXECUTABLE_OUTPUT_PATH})

IF (UNIX)
    IF (NOT CMAKE_BUILD_TYPE) 
        SET(SCHISM_LIBRARY_DIR    ${LIBRARY_OUTPUT_PATH}/release)
        SET(SCHISM_EXECUTABLE_DIR ${EXECUTABLE_OUTPUT_PATH}/release)
    ELSE (NOT CMAKE_BUILD_TYPE) 
        IF (CMAKE_BUILD_TYPE STREQUAL "Release") 
            SET(SCHISM_LIBRARY_DIR    ${LIBRARY_OUTPUT_PATH}/release) 
             SET(SCHISM_EXECUTABLE_DIR ${EXECUTABLE_OUTPUT_PATH}/release) 
        ENDIF (CMAKE_BUILD_TYPE STREQUAL "Release") 
        IF (CMAKE_BUILD_TYPE STREQUAL "Debug") 
            SET(SCHISM_LIBRARY_DIR    ${LIBRARY_OUTPUT_PATH}/debug) 
            SET(SCHISM_EXECUTABLE_DIR ${EXECUTABLE_OUTPUT_PATH}/debug) 
        ENDIF (CMAKE_BUILD_TYPE STREQUAL "Debug") 
    ENDIF (NOT CMAKE_BUILD_TYPE) 
ENDIF (UNIX)


# Include subprojects
ADD_SUBDIRECTORY(core)
ADD_SUBDIRECTORY(ogl)
ADD_SUBDIRECTORY(data)
ADD_SUBDIRECTORY(input)
