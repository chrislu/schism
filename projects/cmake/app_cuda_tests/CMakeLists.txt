
PROJECT(app_cuda_tests)

INCLUDE(${CMAKE_SOURCE_DIR}/cmake_custom_scripts/cuda.cmake)


SET(SRC_DIR "${GLOBAL_SRC_DIR}/app_cuda_tests")

SET(EXECUTABLE_OUTPUT_PATH              ${SCHISM_EXECUTABLE_DIR})

# source files
FILE(GLOB SOURCE_FILES                  ${SRC_DIR}/*.cpp)
FILE(GLOB HEADER_FILES                  ${SRC_DIR}/*.h
                                        ${SRC_DIR}/*.inl)

FILE(GLOB CUDA_SRC_FILES                ${SRC_DIR}/cuda/*.cu
                                        ${SRC_DIR}/cuda/*.h)

FILE(GLOB SHADER_FILES                  ${SRC_DIR}/shader/*.glsl)

# define source groups
SET(SOURCE_GROUP_DELIMITER "/")

SOURCE_GROUP(source_files                   FILES ${SOURCE_FILES}                   ${HEADER_FILES})
SOURCE_GROUP(source_files/cuda              FILES ${GLOB_CUDA_SRC_FILES}            ${CUDA_SRC_FILES})
SOURCE_GROUP(source_files/shader_files      FILES ${SHADER_FILES})

# combine source file lists
SET(SOURCE_FILES
    ${SOURCE_FILES}
)

SET(HEADER_FILES
    ${HEADER_FILES}
    ${CUDA_SRC_FILES}
)

SET(SHADER_FILES
    ${SHADER_FILES}
)

# include header and inline files in source files for visual studio projects
IF (WIN32)
    IF (MSVC)
        SET (SOURCE_FILES ${SOURCE_FILES} ${HEADER_FILES} ${SHADER_FILES})
    ENDIF (MSVC)
ENDIF (WIN32)

# set include directories
INCLUDE_DIRECTORIES(
    ${GLOBAL_SRC_DIR}/app_cuda_tests
    ${GLOBAL_SRC_DIR}/schism
)
CUDA_INCLUDE_DIRECTORIES(
    ${GLOBAL_SRC_DIR}/app_cuda_tests
    ${GLOBAL_SRC_DIR}/schism
)
# set include directories for windows external dependencies
IF (WIN32)
    INCLUDE_DIRECTORIES(
        ${GLOBAL_EXT_DIR}/inc
        ${GLOBAL_EXT_DIR}/inc/boost
        ${GLOBAL_EXT_DIR}/inc/cuda
    )
    CUDA_INCLUDE_DIRECTORIES(
        ${GLOBAL_EXT_DIR}/inc
        ${GLOBAL_EXT_DIR}/inc/boost
        ${GLOBAL_EXT_DIR}/inc/cuda
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

CUDA_ADD_NVCC_COMMANDS(${SOURCE_FILES})

# add/create library
ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCE_FILES})

# link libraries
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
    scm_ogl
    scm_core
)
IF (WIN32)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}
        cuda
        cudart
        glew32
        glut32
        optimized libboost_program_options-${SCHISM_BOOST_LIB_POSTFIX_REL}      debug libboost_program_options-${SCHISM_BOOST_LIB_POSTFIX_DBG}
    )
ELSEIF (UNIX)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}
        GLEW
        glut
        boost_program_options
    )
ENDIF (WIN32)

ADD_DEPENDENCIES(${PROJECT_NAME}
    scm_ogl
    scm_core
)
