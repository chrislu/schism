#TODO
#- copy dll to files to bin directory
#- copy libs to lib directory


#Project Name
PROJECT(schism)

#-----------------------------------------------------------------------------
# Setup schism_VR version strings
SET(SCHISM_MAJOR_VERSION 0)
SET(SCHISM_MINOR_VERSION 0)
SET(SCHISM_BUILD_VERSION 1)
SET(SCHISM_VERSION "${schism_MAJOR_VERSION}.${schism_MINOR_VERSION}.${schism_BUILD_VERSION}")

#-----------------------------------------------------------------------------
# Setup external library strings
SET(SCHISM_BOOST_VERSION            "1_34_1")
SET(SCHISM_BOOST_COMPILER_VERSION   "vc80")
SET(SCHISM_BOOST_LIB_POSTFIX_REL    "${SCHISM_BOOST_COMPILER_VERSION}-mt-${SCHISM_BOOST_VERSION}")
SET(SCHISM_BOOST_LIB_POSTFIX_DBG    "${SCHISM_BOOST_COMPILER_VERSION}-mt-gd-${SCHISM_BOOST_VERSION}")

#-----------------------------------------------------------------------------
# Setup some CMake build options
OPTION(BUILD_PLACE_HOLDER "Build the place_holder Application" ON)


#-----------------------------------------------------------------------------
# Set some directory strings
SET(PLATFORM_WIN32 "win_x86")
SET(PLATFORM_WIN64 "win_x64")
SET(PLATFORM_LINUX32 "linux_x86")
SET(PLATFORM_LINUX64 "linux_x64")

SET(SCHISM_PLATFORM "unsupported")

IF (WIN32)
    IF (MSVC_IDE)
        IF (CMAKE_CL_64)
            SET(SCHISM_PLATFORM ${PLATFORM_WIN64})
        ELSE (CMAKE_CL_64)
            SET(SCHISM_PLATFORM ${PLATFORM_WIN32})
        ENDIF (CMAKE_CL_64)
    ELSE (MSVC_IDE)
        MESSAGE(FATAL_ERROR "unsupported platform")
    ENDIF (MSVC_IDE)
ENDIF (WIN32)

IF (UNIX)
# TODO detection of exact platform!
    SET(SCHISM_PLATFORM ${PLATFORM_LINUX32})
#MESSAGE(FATAL_ERROR "TODO Unix platform stuff")
ENDIF (UNIX)

#-----------------------------------------------------------------------------
# Output directories.
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

# schism_SOURCE_DIR - location of root CMakeLists.txt, i.e. root/projects/cmake
SET(GLOBAL_BIN_DIR ${schism_SOURCE_DIR}/../../bin)
SET(GLOBAL_EXT_DIR ${schism_SOURCE_DIR}/../../ext)
SET(GLOBAL_LIB_DIR ${schism_SOURCE_DIR}/../../lib)
SET(GLOBAL_RES_DIR ${schism_SOURCE_DIR}/../../res)
SET(GLOBAL_SRC_DIR ${schism_SOURCE_DIR}/../../src)

#-----------------------------------------------------------------------------
# Macro definitions

#-----------------------------------------------------------------------------
# Some platfom specific compiler flags
IF (WIN32)
    IF (MSVC80)
        ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE)
        ADD_DEFINITIONS(-D_CRT_NONSTDC_NO_DEPRECATE)
        ADD_DEFINITIONS(-D_SCL_SECURE_NO_DEPRECATE)
        ADD_DEFINITIONS(-DBOOST_ALL_NO_LIB)
        
        SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /D_SECURE_SCL=0")
    ENDIF (MSVC80)
ENDIF (WIN32)

ADD_DEFINITIONS(-DBOOST_DATE_TIME_POSIX_TIME_STD_CONFIG)
ADD_DEFINITIONS(-DSCM_RESSOURCE_DIR="\\"${GLOBAL_RES_DIR}\\"")

#MESSAGE(${CMAKE_SHARED_LIBRARY_SUFFIX})

# Include subprojects
SUBDIRS(
    app_basic_volume_renderer
    app_geometry_correction
    app_cuda_tests
    app_glut_tests
    app_qt4_tests
    app_volume_converter
    schism
)

