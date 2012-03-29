
# Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
# Distributed under the Modified BSD License, see license.txt.

PROJECT(SCHISM_BUILD_ALL_PROJECTS)

# include platform definitions

# find all projects in projects directory
FILE(GLOB SCHISM_SUB_PROJECTS        ${PROJECT_SOURCE_DIR}/../../*/build/cmake)
FILE(GLOB SCHISM_SUB_PROJECTS_SUBDIR ${PROJECT_SOURCE_DIR}/../../*/*/build/cmake)

# retrieve project names and add a build option for them
FOREACH(PROJ_PATH ${SCHISM_SUB_PROJECTS})
    FILE(TO_CMAKE_PATH ${PROJ_PATH} PROJ_PATH)

    # extract project name
    STRING(REGEX REPLACE ".*cmake/+\\.\\./\\.\\./([^/]+)/.*" "\\1" PROJ_NAME ${PROJ_PATH})
    # add build option
    OPTION(BUILD_${PROJ_NAME} "Build ${PROJ_NAME}" OFF)

    # add project subdirectory
    IF (BUILD_${PROJ_NAME})
        ADD_SUBDIRECTORY(${PROJ_PATH} ${CMAKE_CURRENT_BINARY_DIR}/${PROJ_NAME})
    ENDIF (BUILD_${PROJ_NAME})
ENDFOREACH(PROJ_PATH)

# retrieve project names and add a build option for them
FOREACH(PROJ_PATH ${SCHISM_SUB_PROJECTS_SUBDIR})
    FILE(TO_CMAKE_PATH ${PROJ_PATH} PROJ_PATH)

    # extract project name
    STRING(REGEX REPLACE ".*cmake/+\\.\\./\\.\\./([^/]+)/([^/]+)/.*" "\\1" PROJ_MASTER ${PROJ_PATH})
    STRING(REGEX REPLACE ".*cmake/+\\.\\./\\.\\./([^/]+)/([^/]+)/.*" "\\2" PROJ_NAME   ${PROJ_PATH})
    # add build option
    OPTION(BUILD_${PROJ_MASTER}_${PROJ_NAME} "Build ${PROJ_NAME}" OFF)

    # add project subdirectory
    IF (BUILD_${PROJ_MASTER}_${PROJ_NAME})
        ADD_SUBDIRECTORY(${PROJ_PATH} ${CMAKE_CURRENT_BINARY_DIR}/${PROJ_MASTER}/${PROJ_NAME})
    ENDIF (BUILD_${PROJ_MASTER}_${PROJ_NAME})
ENDFOREACH(PROJ_PATH)
