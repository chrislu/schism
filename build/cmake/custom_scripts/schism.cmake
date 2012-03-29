
# Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
# Distributed under the Modified BSD License, see license.txt.

option(SCHISM_BUILD_STATIC	"Build schism libraries as static libraries" ON)

#-----------------------------------------------------------------------------
# Setup schism version strings
set(SCHISM_MAJOR_VERSION 0)
set(SCHISM_MINOR_VERSION 7)
set(SCHISM_BUILD_VERSION 9)
set(SCHISM_VERSION "${schism_MAJOR_VERSION}.${schism_MINOR_VERSION}.${schism_BUILD_VERSION}")

#-----------------------------------------------------------------------------
# setup output directories.
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${schism_SOURCE_DIR}/../../lib/${SCHISM_PLATFORM} CACHE INTERNAL "Single output directory for building all libraries.")
endif(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)

if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${schism_SOURCE_DIR}/../../lib/${SCHISM_PLATFORM} CACHE INTERNAL "Single output directory for building all libraries.")
endif(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)

if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${schism_SOURCE_DIR}/../../bin/${SCHISM_PLATFORM} CACHE INTERNAL "Single output directory for building all executables.")
endif(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)

set(SCHISM_LIBRARY_DIR    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(SCHISM_EXECUTABLE_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# schism_SOURCE_DIR - location of root CMakeLists.txt, i.e. root/schism/build/cmake
set(SCM_BIN_DIR ${schism_SOURCE_DIR}/../../bin)
set(SCM_LIB_DIR ${schism_SOURCE_DIR}/../../lib)
set(SCM_ROOT_DIR ${schism_SOURCE_DIR}/../..)

if (SCHISM_BUILD_STATIC)
	add_definitions(-DSCM_STATIC_BUILD)
endif (SCHISM_BUILD_STATIC)
