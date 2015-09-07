
# Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
# Distributed under the Modified BSD License, see license.txt.

SET(SCM_QT4_MIN_VERSION_MAJOR 4)
SET(SCM_QT4_MIN_VERSION_MINOR 6)
SET(SCM_QT4_MIN_VERSION_SUBMINOR 0)
SET(SCM_QT4_MIN_VERSION "${SCM_QT4_MIN_VERSION_MAJOR}.${SCM_QT4_MIN_VERSION_MINOR}.${SCM_QT4_MIN_VERSION_SUBMINOR}")

SET(SCM_QT4_INCLUDE_SEARCH_DIRS
    ${GLOBAL_EXT_DIR}/inc/qt4
    ${GLOBAL_EXT_DIR}/inc/qt
    ${CMAKE_SYSTEM_INCLUDE_PATH}
    ${CMAKE_INCLUDE_PATH}
    /usr/include/qt4
)

SET(SCM_QT4_LIBRARY_SEARCH_DIRS
    ${GLOBAL_EXT_DIR}/lib
    ${CMAKE_SYSTEM_LIBRARY_PATH}
    ${CMAKE_LIBRARY_PATH}
    /usr/lib64
)

SET(SCM_QT4_BINARY_SEARCH_DIRS
    ${GLOBAL_EXT_DIR}/bin/qt4
    ${GLOBAL_EXT_DIR}/bin/qt
    $ENV{PATH}
    $ENV{QTDIR}/bin
)

IF (NOT SCM_QT4_INC_DIR)

    SET(_SCM_QT4_FOUND_INC_DIRS "")

    FOREACH(_SEARCH_DIR ${SCM_QT4_INCLUDE_SEARCH_DIRS})
        FIND_PATH(_CUR_SEARCH
                NAMES QtCore/qglobal.h
                PATHS ${_SEARCH_DIR}
                NO_DEFAULT_PATH)
        IF (_CUR_SEARCH)
            #MESSAGE(${_CUR_SEARCH})
            LIST(APPEND _SCM_QT4_FOUND_INC_DIRS ${_CUR_SEARCH})
        ENDIF(_CUR_SEARCH)
        SET(_CUR_SEARCH _CUR_SEARCH-NOTFOUND CACHE INTERNAL "internal use")
    ENDFOREACH(_SEARCH_DIR ${SCM_QT4_INCLUDE_SEARCH_DIRS})

    IF (NOT _SCM_QT4_FOUND_INC_DIRS)
        MESSAGE(FATAL_ERROR "schism_qt4.cmake: unable to find qt library")
    ENDIF (NOT _SCM_QT4_FOUND_INC_DIRS)

    #MESSAGE(${_SCM_QT4_FOUND_INC_DIRS})

    SET(SCM_QT4_VERSION       0)

    MATH(EXPR SCM_QT4_MIN_VERSION_NUM "${SCM_QT4_MIN_VERSION_MAJOR}*10000 + ${SCM_QT4_MIN_VERSION_MINOR}*100 + ${SCM_QT4_MIN_VERSION_SUBMINOR}")
    SET(_SCM_QT4_CUR_VERSION ${SCM_QT4_MIN_VERSION_NUM})

    FOREACH(_INC_DIR ${_SCM_QT4_FOUND_INC_DIRS})
        FILE(READ "${_INC_DIR}/QtCore/qglobal.h" _SCM_QT4_VERSION_CONTENTS)

        STRING(REGEX REPLACE ".*#define QT_VERSION_STR [ ]*\"([0-9]+)\\.[0-9]+\\.[0-9]+[-0-9a-zA-Z]*\".*" "\\1" _QT_VERSION_MAJOR     ${_SCM_QT4_VERSION_CONTENTS})
        STRING(REGEX REPLACE ".*#define QT_VERSION_STR [ ]*\"[0-9]+\\.([0-9]+)\\.[0-9]+[-0-9a-zA-Z]*\".*" "\\1" _QT_VERSION_MINOR     ${_SCM_QT4_VERSION_CONTENTS})
        STRING(REGEX REPLACE ".*#define QT_VERSION_STR [ ]*\"[0-9]+\\.[0-9]+\\.([0-9]+)[-0-9a-zA-Z]*\".*" "\\1" _QT_VERSION_SUBMINOR  ${_SCM_QT4_VERSION_CONTENTS})

        MESSAGE(STATUS "Found Qt4 version: ${_QT_VERSION_MAJOR}.${_QT_VERSION_MINOR}.${_QT_VERSION_SUBMINOR}")

        MATH(EXPR _QT_VERSION_NUM     "${_QT_VERSION_MAJOR}*10000 + ${_QT_VERSION_MINOR}*100 + ${_QT_VERSION_SUBMINOR}")

        #MESSAGE(${_QT_VERSION_NUM})

        IF (   _SCM_QT4_CUR_VERSION LESS  _QT_VERSION_NUM
            OR _SCM_QT4_CUR_VERSION EQUAL _QT_VERSION_NUM)
            SET(SCM_QT4_VERSION          ${_QT_VERSION_NUM})
            SET(SCM_QT4_LIB_SUFFIX       ".${_QT_VERSION_MAJOR}.${_QT_VERSION_MINOR}.${_QT_VERSION_SUBMINOR}")
            SET(SCM_QT4_INC_DIR          ${_INC_DIR})
            SET(_SCM_QT4_CUR_VERSION     ${_QT_VERSION_NUM})
        ENDIF (   _SCM_QT4_CUR_VERSION LESS  _QT_VERSION_NUM
               OR _SCM_QT4_CUR_VERSION EQUAL _QT_VERSION_NUM)

    ENDFOREACH(_INC_DIR ${_SCM_QT4_FOUND_INC_DIRS})

    IF (SCM_QT4_VERSION EQUAL 0)
        MESSAGE(FATAL_ERROR "found qt versions ${_QT_VERSION_NUM} to old (min. version ${SCM_QT4_MIN_VERSION} required)")
    ELSE (SCM_QT4_VERSION EQUAL 0)
        SET(SCM_QT4_INC_DIR          ${SCM_QT4_INC_DIR}           CACHE STRING "The qt include directory")
        SET(SCM_QT4_VERSION          ${SCM_QT4_VERSION}           CACHE STRING "The qt version number")
        SET(SCM_QT4_LIB_SUFFIX       ${SCM_QT4_LIB_SUFFIX}        CACHE STRING "The qt library suffix")
    ENDIF (SCM_QT4_VERSION EQUAL 0)
ENDIF (NOT SCM_QT4_INC_DIR)

IF (        SCM_QT4_INC_DIR
    AND NOT SCM_QT4_LIB_DIR)

    LIST(APPEND _SCM_QT4_CORE_LIB    "${CMAKE_SHARED_LIBRARY_PREFIX}QtCore${CMAKE_SHARED_LIBRARY_SUFFIX}${SCM_QT4_LIB_SUFFIX}"
                                    "${CMAKE_SHARED_LIBRARY_PREFIX}QtCore4${CMAKE_SHARED_LIBRARY_SUFFIX}")

    #MESSAGE(${_SCM_QT4_CORE_LIB})

    SET(_SCM_QT4_FOUND_LIB_DIR "")

    FOREACH(_SEARCH_DIR ${SCM_QT4_LIBRARY_SEARCH_DIRS})
        #MESSAGE(${_SEARCH_DIR})
        FIND_PATH(_CUR_SEARCH
                NAMES ${_SCM_QT4_CORE_LIB}
                PATHS ${_SEARCH_DIR}
                PATH_SUFFIXES debug release
                NO_DEFAULT_PATH)
        IF (_CUR_SEARCH)
            #MESSAGE(${_CUR_SEARCH})
            LIST(APPEND _SCM_QT4_FOUND_LIB_DIR ${_SEARCH_DIR})
        ENDIF(_CUR_SEARCH)
        SET(_CUR_SEARCH _CUR_SEARCH-NOTFOUND CACHE INTERNAL "internal use")
    ENDFOREACH(_SEARCH_DIR ${SCM_QT4_LIBRARY_SEARCH_DIRS})

    #MESSAGE(${_SCM_QT4_FOUND_LIB_DIR})

    IF (NOT _SCM_QT4_FOUND_LIB_DIR)
        MESSAGE(FATAL_ERROR "schism_qt4.cmake: unable to find qt library")
    ELSE (NOT _SCM_QT4_FOUND_LIB_DIR)
        SET(SCM_QT4_LIB_DIR          ${_SCM_QT4_FOUND_LIB_DIR}        CACHE STRING "The qt library directory")
    ENDIF (NOT _SCM_QT4_FOUND_LIB_DIR)

ENDIF(        SCM_QT4_INC_DIR
      AND NOT SCM_QT4_LIB_DIR)

IF (   NOT SCM_QT4_MOC_EXECUTABLE
    OR NOT SCM_QT4_RCC_EXECUTABLE
    OR NOT SCM_QT4_UIC_EXECUTABLE)

    #FIND_PROGRAM(SCM_QT4_MOC_EXECUTABLE "moc" PATHS ${GLOBAL_EXT_DIR}/bin/qt $ENV{PATH} $ENV{QTDIR}/bin)
    #FIND_PROGRAM(SCM_QT4_RCC_EXECUTABLE "rcc" PATHS ${GLOBAL_EXT_DIR}/bin/qt $ENV{PATH} $ENV{QTDIR}/bin)
    #FIND_PROGRAM(SCM_QT4_UIC_EXECUTABLE "uic" PATHS ${GLOBAL_EXT_DIR}/bin/qt $ENV{PATH} $ENV{QTDIR}/bin)

    FIND_PROGRAM(SCM_QT4_MOC_EXECUTABLE NAMES moc-qt4 moc PATHS ${SCM_QT4_BINARY_SEARCH_DIRS})
    FIND_PROGRAM(SCM_QT4_RCC_EXECUTABLE NAMES rcc         PATHS ${SCM_QT4_BINARY_SEARCH_DIRS})
    FIND_PROGRAM(SCM_QT4_UIC_EXECUTABLE NAMES uic-qt4 uic PATHS ${SCM_QT4_BINARY_SEARCH_DIRS})

    IF (NOT SCM_QT4_MOC_EXECUTABLE)
        MESSAGE(FATAL_ERROR "unable to find qt4 moc")
    ENDIF (NOT SCM_QT4_MOC_EXECUTABLE)
    IF (NOT SCM_QT4_RCC_EXECUTABLE)
        MESSAGE(FATAL_ERROR "unable to find qt4 rcc")
    ENDIF (NOT SCM_QT4_RCC_EXECUTABLE)
    IF (NOT SCM_QT4_UIC_EXECUTABLE)
        MESSAGE(FATAL_ERROR "unable to find qt4 uic")
    ENDIF (NOT SCM_QT4_UIC_EXECUTABLE)

    EXEC_PROGRAM(${SCM_QT4_UIC_EXECUTABLE} ARGS "-version" OUTPUT_VARIABLE TMP_UIC_OUTPUT)

    STRING(REGEX REPLACE ".*version ([0-9]+\\.[0-9]+\\.[0-9]+).*" "\\1" _QT_VERSION ${TMP_UIC_OUTPUT})

    STRING(REGEX REPLACE "([0-9]+)\\.[0-9]+\\.[0-9]+" "\\1" _QT_VERSION_MAJOR ${_QT_VERSION})
    STRING(REGEX REPLACE "[0-9]+\\.([0-9]+)\\.[0-9]+" "\\1" _QT_VERSION_MINOR ${_QT_VERSION})
    STRING(REGEX REPLACE "[0-9]+\\.[0-9]+\\.([0-9]+)" "\\1" _QT_VERSION_BUILD ${_QT_VERSION})

    MATH(EXPR _QT_VERSION_NUM     "${_QT_VERSION_MAJOR}*10000        + ${_QT_VERSION_MINOR}*100        + ${_QT_VERSION_BUILD}")

    IF (SCM_QT4_VERSION EQUAL _QT_VERSION_NUM)
        SET(SCM_QT4_MOC_EXECUTABLE ${SCM_QT4_MOC_EXECUTABLE}      CACHE STRING "The qt moc binary")
        SET(SCM_QT4_RCC_EXECUTABLE ${SCM_QT4_RCC_EXECUTABLE}      CACHE STRING "The qt rcc binary")
        SET(SCM_QT4_UIC_EXECUTABLE ${SCM_QT4_UIC_EXECUTABLE}      CACHE STRING "The qt uic binary")
    ELSE (SCM_QT4_VERSION EQUAL _QT_VERSION_NUM)
        MESSAGE(FATAL_ERROR "schism_qt4.cmake: found qt binaries version ${_QT_VERSION_NUM} unequal found library (version ${SCM_QT4_VERSION} required)")
    ENDIF (SCM_QT4_VERSION EQUAL _QT_VERSION_NUM)

ENDIF(   NOT SCM_QT4_MOC_EXECUTABLE
      OR NOT SCM_QT4_RCC_EXECUTABLE
      OR NOT SCM_QT4_UIC_EXECUTABLE)

macro(scm_add_moc_command SCM_MOC_OUTPUT_FILES)
    set(input_file_list  ${ARGV})
    set(output_file_list )
    # remove the leading output variable
    list(REMOVE_AT input_file_list 0)

    FOREACH(input_file ${input_file_list})
        get_filename_component(input_file           ${input_file} ABSOLUTE)
        get_filename_component(input_file_name      ${input_file} NAME)
        get_filename_component(input_file_name_we   ${input_file} NAME_WE)
        get_filename_component(input_file_path      ${input_file} PATH)
        get_filename_component(project_src_path     ${SCM_PROJECT_SOURCE_DIR} ABSOLUTE)

        set(output_file_path "${input_file_path}/_moc")
        file(MAKE_DIRECTORY ${output_file_path})
        set(output_file  "${output_file_path}/${input_file_name_we}_moc.cpp")
        list(APPEND output_file_list ${output_file})

        file(RELATIVE_PATH project_path ${project_src_path} ${output_file_path})
        if (project_path)
            file(TO_CMAKE_PATH ${project_path} project_path)
        endif (project_path)

        source_group(source_files/${project_path} FILES ${output_file})

        add_custom_command(OUTPUT   ${output_file}
                           COMMAND  ${SCM_QT4_MOC_EXECUTABLE}
                               ARGS -o \"${output_file}\"
                                    \"${input_file}\"
        #                   MAIN_DEPENDENCY ${input_file}
                           DEPENDS  ${input_file}
                           COMMENT "moc ${input_file_name}")

    endforeach(input_file)

    #set(${SCM_MOC_OUTPUT_FILES} ${output_file_list})
    set_source_files_properties(${output_file_list} PROPERTIES GENERATED TRUE)
    list(APPEND ${SCM_MOC_OUTPUT_FILES} ${output_file_list})

ENDMACRO(scm_add_moc_command SCM_MOC_OUTPUT_FILES)

# important when qt is used along with boost::signals
ADD_DEFINITIONS(-DQT_NO_KEYWORDS)
