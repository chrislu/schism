
# Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
# Distributed under the Modified BSD License, see license.txt.

#-----------------------------------------------------------------------------
# Set some directory strings
SET(PLATFORM_WIN32      "win_x86")
SET(PLATFORM_WIN64      "win_x64")
SET(PLATFORM_LINUX32    "linux_x86")
SET(PLATFORM_LINUX64    "linux_x64")

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

#IF (UNIX)
#    IF (NOT CMAKE_BUILD_TYPE)
#        SET(SCHISM_LIBRARY_DIR    ${LIBRARY_OUTPUT_PATH}/release)
#        SET(SCHISM_EXECUTABLE_DIR ${EXECUTABLE_OUTPUT_PATH}/release)
#    ELSE (NOT CMAKE_BUILD_TYPE)
#        IF (CMAKE_BUILD_TYPE STREQUAL "Release")
#            SET(SCHISM_LIBRARY_DIR    ${LIBRARY_OUTPUT_PATH}/release)
#             SET(SCHISM_EXECUTABLE_DIR ${EXECUTABLE_OUTPUT_PATH}/release)
#        ENDIF (CMAKE_BUILD_TYPE STREQUAL "Release")
#        IF (CMAKE_BUILD_TYPE STREQUAL "Debug")
#            SET(SCHISM_LIBRARY_DIR    ${LIBRARY_OUTPUT_PATH}/debug)
#            SET(SCHISM_EXECUTABLE_DIR ${EXECUTABLE_OUTPUT_PATH}/debug)
#        ENDIF (CMAKE_BUILD_TYPE STREQUAL "Debug")
#    ENDIF (NOT CMAKE_BUILD_TYPE)
#ENDIF (UNIX)

#-----------------------------------------------------------------------------
# Some platfom specific compiler flags
IF (WIN32)
    IF (MSVC80 OR MSVC90)
        ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE)
        ADD_DEFINITIONS(-D_CRT_NONSTDC_NO_DEPRECATE)
        ADD_DEFINITIONS(-D_SCL_SECURE_NO_DEPRECATE)
        ADD_DEFINITIONS(-DBOOST_ALL_NO_LIB)

        SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /D_SECURE_SCL=0")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    ENDIF (MSVC80 OR MSVC90)
    if (MSVC10 OR MSVC11)
        add_definitions(-D_SCL_SECURE_NO_WARNINGS)
        add_definitions(-D_CRT_SECURE_NO_WARNINGS)
        add_definitions(-DBOOST_ALL_NO_LIB)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    endif (MSVC10 OR MSVC11)
    ADD_DEFINITIONS(-DNOMINMAX)
ENDIF (WIN32)

if (UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif (UNIX)

#-----------------------------------------------------------------------------
# enable nanosec precision of boost::date_time library
add_definitions(-DBOOST_DATE_TIME_POSIX_TIME_STD_CONFIG)

if (UNIX)
    include_directories(/opt/OpenCL/include
                        /opt/cuda/current/cuda/include)
    link_directories(/opt/cuda/current/cuda/lib64
                     /usr/lib/nvidia-current)
endif (UNIX)
