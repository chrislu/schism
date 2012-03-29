
# Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
# Distributed under the Modified BSD License, see license.txt.

INCLUDE(schism_compiler)

SET(SCM_BOOST_MIN_VERSION_MAJOR 1)
SET(SCM_BOOST_MIN_VERSION_MINOR 46)
SET(SCM_BOOST_MIN_VERSION_SUBMINOR 0)
SET(SCM_BOOST_MIN_VERSION "${SCM_BOOST_MIN_VERSION_MAJOR}.${SCM_BOOST_MIN_VERSION_MINOR}.${SCM_BOOST_MIN_VERSION_SUBMINOR}")
MATH(EXPR SCM_BOOST_MIN_VERSION_NUM "${SCM_BOOST_MIN_VERSION_MAJOR}*10000 + ${SCM_BOOST_MIN_VERSION_MINOR}*100 + ${SCM_BOOST_MIN_VERSION_SUBMINOR}")

SET(SCM_BOOST_INCLUDE_SEARCH_DIRS
    ${GLOBAL_EXT_DIR}/inc/boost
    /opt/boost/default/include
    ${CMAKE_SYSTEM_INCLUDE_PATH}
    ${CMAKE_INCLUDE_PATH}
    /usr/include
)

SET(SCM_BOOST_LIBRARY_SEARCH_DIRS
    ${GLOBAL_EXT_DIR}/lib
    /opt/boost/default/lib
    ${CMAKE_SYSTEM_LIBRARY_PATH}
    ${CMAKE_LIBRARY_PATH}
    /usr/lib64
)

IF (NOT SCM_BOOST_INC_DIR)

    SET(_SCM_BOOST_FOUND_INC_DIRS "")

    FOREACH(_SEARCH_DIR ${SCM_BOOST_INCLUDE_SEARCH_DIRS})
        FIND_PATH(_CUR_SEARCH
                NAMES boost/config.hpp
                PATHS ${_SEARCH_DIR}
                NO_DEFAULT_PATH)
        IF (_CUR_SEARCH)
            #MESSAGE(${_CUR_SEARCH})
            LIST(APPEND _SCM_BOOST_FOUND_INC_DIRS ${_CUR_SEARCH})
        ENDIF(_CUR_SEARCH)
        SET(_CUR_SEARCH _CUR_SEARCH-NOTFOUND CACHE INTERNAL "internal use")
    ENDFOREACH(_SEARCH_DIR ${SCM_BOOST_INCLUDE_SEARCH_DIRS})

    IF (NOT _SCM_BOOST_FOUND_INC_DIRS)
        MESSAGE(FATAL_ERROR "schism_boost.cmake: unable to find boost library")
    ENDIF (NOT _SCM_BOOST_FOUND_INC_DIRS)

    SET(SCM_BOOST_VERSION       0)
    SET(SCM_BOOST_LIB_VERSION   "")
    SET(SCM_BOOST_LIB_SUFFIX    "")

    SET(_SCM_BOOST_CUR_VERSION ${SCM_BOOST_MIN_VERSION_NUM})

    FOREACH(_INC_DIR ${_SCM_BOOST_FOUND_INC_DIRS})
        FILE(READ "${_INC_DIR}/boost/version.hpp" _SCM_BOOST_VERSION_CONTENTS)

        STRING(REGEX REPLACE ".*#define BOOST_VERSION ([0-9]+).*" "\\1"             _BOOST_VERSION       "${_SCM_BOOST_VERSION_CONTENTS}")
        STRING(REGEX REPLACE ".*#define BOOST_LIB_VERSION \"([0-9_]+)\".*" "\\1"    _BOOST_LIB_VERSION   "${_SCM_BOOST_VERSION_CONTENTS}")

        IF(NOT "${_BOOST_VERSION}" STREQUAL "0")
            MATH(EXPR _BOOST_MAJOR_VERSION      "${_BOOST_VERSION} / 100000")
            MATH(EXPR _BOOST_MINOR_VERSION      "${_BOOST_VERSION} / 100 % 1000")
            MATH(EXPR _BOOST_SUBMINOR_VERSION   "${_BOOST_VERSION} % 100")
        ELSE (NOT "${_BOOST_VERSION}" STREQUAL "0")
            MESSAGE("WTF")
        ENDIF(NOT "${_BOOST_VERSION}" STREQUAL "0")

        MATH(EXPR _BOOST_VERSION_NUM    "${_BOOST_MAJOR_VERSION}*10000  + ${_BOOST_MINOR_VERSION}*100   + ${_BOOST_SUBMINOR_VERSION}")

        IF (   _SCM_BOOST_CUR_VERSION LESS  _BOOST_VERSION_NUM
            OR _SCM_BOOST_CUR_VERSION EQUAL _BOOST_VERSION_NUM)
            SET(SCM_BOOST_VERSION               ${_BOOST_VERSION})
            SET(SCM_BOOST_LIB_VERSION           ${_BOOST_LIB_VERSION})
            SET(SCM_BOOST_LIB_SUFFIX            ".${_BOOST_MAJOR_VERSION}.${_BOOST_MINOR_VERSION}.${_BOOST_SUBMINOR_VERSION}")
            SET(SCM_BOOST_INC_DIR               ${_INC_DIR})
            SET(_SCM_BOOST_CUR_VERSION          ${_BOOST_VERSION_NUM})
        ENDIF (   _SCM_BOOST_CUR_VERSION LESS  _BOOST_VERSION_NUM
               OR _SCM_BOOST_CUR_VERSION EQUAL _BOOST_VERSION_NUM)

    ENDFOREACH(_INC_DIR ${_SCM_BOOST_FOUND_INC_DIRS})

    IF (SCM_BOOST_VERSION EQUAL 0)
        MESSAGE(FATAL_ERROR "found boost versions ${_BOOST_VERSION} to old (min. version ${SCM_BOOST_MIN_VERSION} required)")
    ELSE (SCM_BOOST_VERSION EQUAL 0)
        SET(SCM_BOOST_INC_DIR               ${SCM_BOOST_INC_DIR}               CACHE STRING "The boost include directory")
        SET(SCM_BOOST_VERSION               ${SCM_BOOST_VERSION}               CACHE STRING "The boost version number")
        SET(SCM_BOOST_LIB_SUFFIX            ${SCM_BOOST_LIB_SUFFIX}            CACHE STRING "The boost library suffix")
        SET(SCM_BOOST_LIB_VERSION           ${SCM_BOOST_LIB_VERSION}           CACHE STRING "The boost library version string")
    ENDIF (SCM_BOOST_VERSION EQUAL 0)

ENDIF(NOT SCM_BOOST_INC_DIR)

IF (        SCM_BOOST_INC_DIR
    AND NOT SCM_BOOST_LIB_DIR)

    SET(_SCM_BOOST_FILESYSTEM_LIB "")
    SET(_SCM_BOOST_FOUND_LIB_DIR "")
    SET(_SCM_BOOST_POSTFIX "")

    if (UNIX)
        LIST(APPEND _SCM_BOOST_FILESYSTEM_LIB   "${CMAKE_SHARED_LIBRARY_PREFIX}boost_filesystem-${SCM_COMPILER_SUFFIX}-mt-${SCM_BOOST_LIB_VERSION}${CMAKE_SHARED_LIBRARY_SUFFIX}${SCM_BOOST_LIB_SUFFIX}"
                                                "${CMAKE_SHARED_LIBRARY_PREFIX}boost_filesystem-mt${CMAKE_SHARED_LIBRARY_SUFFIX}${SCM_BOOST_LIB_SUFFIX}"
                                                "${CMAKE_SHARED_LIBRARY_PREFIX}boost_filesystem${CMAKE_SHARED_LIBRARY_SUFFIX}${SCM_BOOST_LIB_SUFFIX}")
    elseif (WIN32)
        LIST(APPEND _SCM_BOOST_FILESYSTEM_LIB   "${CMAKE_SHARED_LIBRARY_PREFIX}boost_filesystem-${SCM_COMPILER_SUFFIX}-mt-${SCM_BOOST_LIB_VERSION}${CMAKE_SHARED_LIBRARY_SUFFIX}"
                                                "${CMAKE_SHARED_LIBRARY_PREFIX}boost_filesystem-mt${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif (UNIX)


    #MESSAGE(${_SCM_BOOST_FILESYSTEM_LIB})

    FOREACH(_SEARCH_DIR ${SCM_BOOST_LIBRARY_SEARCH_DIRS})
        #MESSAGE(${_SEARCH_DIR})
        FIND_PATH(_CUR_SEARCH
                NAMES ${_SCM_BOOST_FILESYSTEM_LIB}
                PATHS ${_SEARCH_DIR}
                PATH_SUFFIXES debug release
                NO_DEFAULT_PATH)
        IF (_CUR_SEARCH)
            #MESSAGE(${_CUR_SEARCH})
            LIST(APPEND _SCM_BOOST_FOUND_LIB_DIR ${_SEARCH_DIR})
            if (UNIX)
                FIND_FILE(_CUR_SEARCH_FILE NAMES ${_SCM_BOOST_FILESYSTEM_LIB} PATHS ${_CUR_SEARCH})
                if (_CUR_SEARCH_FILE)
                    MESSAGE(${_CUR_SEARCH_FILE})
                    STRING(REGEX REPLACE "${_CUR_SEARCH}/${CMAKE_SHARED_LIBRARY_PREFIX}boost_filesystem(.*).so(.*)" "\\2" _SCM_BOOST_UNIX_LIB_SUF ${_CUR_SEARCH_FILE})
                    message(${_SCM_BOOST_UNIX_LIB_SUF})
                    if (${_SCM_BOOST_UNIX_LIB_SUF} STREQUAL ${SCM_BOOST_LIB_SUFFIX})
                        message("found matching version")
                        STRING(REGEX REPLACE "${_CUR_SEARCH}/${CMAKE_SHARED_LIBRARY_PREFIX}boost_filesystem(.*).so(.*)" "\\1" _SCM_BOOST_POSTFIX ${_CUR_SEARCH_FILE})
                        #MESSAGE(${_SCM_BOOST_POSTFIX})
                    endif (${_SCM_BOOST_UNIX_LIB_SUF} STREQUAL ${SCM_BOOST_LIB_SUFFIX})
                endif (_CUR_SEARCH_FILE)
                SET(_CUR_SEARCH_FILE _CUR_SEARCH_FILE-NOTFOUND CACHE INTERNAL "internal use")
            endif (UNIX)
        ENDIF(_CUR_SEARCH)
        SET(_CUR_SEARCH _CUR_SEARCH-NOTFOUND CACHE INTERNAL "internal use")
    ENDFOREACH(_SEARCH_DIR ${SCM_BOOST_LIBRARY_SEARCH_DIRS})

    #MESSAGE(${_SCM_BOOST_FOUND_LIB_DIR})

    IF (NOT _SCM_BOOST_FOUND_LIB_DIR)
        MESSAGE(FATAL_ERROR "schism_boost.cmake: unable to find boost library")
    ELSE (NOT _SCM_BOOST_FOUND_LIB_DIR)
        SET(SCM_BOOST_LIB_DIR       ${_SCM_BOOST_FOUND_LIB_DIR}       CACHE STRING "The boost library directory")
    ENDIF (NOT _SCM_BOOST_FOUND_LIB_DIR)

    if (UNIX)
#            SET(SCHISM_BOOST_LIB_POSTFIX_REL    "${_SCM_BOOST_POSTFIX}"     CACHE STRING "(deprecated) boost library release postfix")
#            SET(SCHISM_BOOST_LIB_POSTFIX_DBG    "${_SCM_BOOST_POSTFIX}"     CACHE STRING "(deprecated) boost library debug postfix")
#        IF (NOT _SCM_BOOST_POSTFIX)
#            MESSAGE(FATAL_ERROR "schism_boost.cmake: unable to determine boost library suffix")
#        ELSE (NOT _SCM_BOOST_POSTFIX)
#            SET(SCHISM_BOOST_LIB_POSTFIX_REL    "${_SCM_BOOST_POSTFIX}"     CACHE STRING "boost library release postfix")
#            SET(SCHISM_BOOST_LIB_POSTFIX_DBG    "${_SCM_BOOST_POSTFIX}"     CACHE STRING "boost library debug postfix")
#        ENDIF (NOT _SCM_BOOST_POSTFIX)
        set(SCM_BOOST_MT_REL   "${_SCM_BOOST_POSTFIX}" CACHE STRING "boost dynamic library release postfix")
        set(SCM_BOOST_MT_DBG   "${_SCM_BOOST_POSTFIX}" CACHE STRING "boost dynamic library debug postfix")
        set(SCM_BOOST_MT_S_REL "${_SCM_BOOST_POSTFIX}" CACHE STRING "boost static library release postfix")
        set(SCM_BOOST_MT_S_DBG "${_SCM_BOOST_POSTFIX}" CACHE STRING "boost static library debug postfix")
    elseif (WIN32)
#        SET(SCHISM_BOOST_LIB_POSTFIX_REL    "${SCM_COMPILER_SUFFIX}-mt-${SCM_BOOST_LIB_VERSION}"     CACHE STRING "(deprecated) boost library release postfix")
#        SET(SCHISM_BOOST_LIB_POSTFIX_DBG    "${SCM_COMPILER_SUFFIX}-mt-gd-${SCM_BOOST_LIB_VERSION}"  CACHE STRING "(deprecated) boost library debug postfix")

        set(SCM_BOOST_MT_REL    "${SCM_COMPILER_SUFFIX}-mt-${SCM_BOOST_LIB_VERSION}"     CACHE STRING "boost dynamic library release postfix")
        set(SCM_BOOST_MT_DBG    "${SCM_COMPILER_SUFFIX}-mt-gd-${SCM_BOOST_LIB_VERSION}"  CACHE STRING "boost dynamic library debug postfix")
        set(SCM_BOOST_MT_S_REL  "${SCM_COMPILER_SUFFIX}-mt-s-${SCM_BOOST_LIB_VERSION}"   CACHE STRING "boost static library release postfix")
        set(SCM_BOOST_MT_S_DBG  "${SCM_COMPILER_SUFFIX}-mt-sgd-${SCM_BOOST_LIB_VERSION}" CACHE STRING "boost static library debug postfix")
    endif (UNIX)

ENDIF(        SCM_BOOST_INC_DIR
      AND NOT SCM_BOOST_LIB_DIR)



