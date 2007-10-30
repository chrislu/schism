
OPTION(CUDA_BUILD_EMULATION                 "Build emulation CUDA code"                                     OFF)
OPTION(CUDA_BUILD_USE_FAST_MATH             "Use lower precision math calculations"                         OFF)
OPTION(CUDA_BUILD_KEEP_INTERMEDIATE_FILES   "Keep the generated intermediate files"                         OFF)
OPTION(CUDA_BUILD_DISPLAY_COMPILE_GPU_INFO  "Display resulting information about register usage and so on"  ON)

SET(CUDA_NVCC_PATH                  "${GLOBAL_EXT_DIR}/bin/cuda/bin")
SET(CUDE_NVCC_COMMAND               "${CUDA_NVCC_PATH}/nvcc")

SET(CUDA_NVCC_CXX_FLAGS             ${CMAKE_CXX_FLAGS})
SET(CUDA_NVCC_CXX_FLAGS_RELEASE     ${CMAKE_CXX_FLAGS_RELEASE})
SET(CUDA_NVCC_CXX_FLAGS_DEBUG       ${CMAKE_CXX_FLAGS_DEBUG})

# convert CXX compiler options into comma separated list
STRING(REPLACE " " "," CUDA_NVCC_CXX_FLAGS          ${CUDA_NVCC_CXX_FLAGS})
STRING(REPLACE " " "," CUDA_NVCC_CXX_FLAGS_RELEASE  ${CUDA_NVCC_CXX_FLAGS_RELEASE})
STRING(REPLACE " " "," CUDA_NVCC_CXX_FLAGS_DEBUG    ${CUDA_NVCC_CXX_FLAGS_DEBUG})

SET(CUDA_NVCC_DEFINITIONS           "")
SET(CUDA_NVCC_INCLUDE_DIRECTORIES   "")

SET(CUDA_NVCC_OPTIONS               "-ccbin \"$(VCInstallDir)bin\"")
SET(CUDA_NVCC_OPTIONS               "${CUDA_NVCC_OPTIONS} -Xcompiler ${CUDA_NVCC_CXX_FLAGS_RELEASE}${CUDA_NVCC_CXX_FLAGS}")

IF (CUDA_BUILD_EMULATION)
    SET(CUDA_NVCC_OPTIONS "${CUDA_NVCC_OPTIONS} -deviceemu -g")
ENDIF(CUDA_BUILD_EMULATION)

IF (CUDA_BUILD_KEEP_INTERMEDIATE_FILES)
    SET(CUDA_NVCC_OPTIONS "${CUDA_NVCC_OPTIONS} --keep")
ENDIF(CUDA_BUILD_KEEP_INTERMEDIATE_FILES)

IF (CUDA_BUILD_USE_FAST_MATH)
    SET(CUDA_NVCC_OPTIONS "${CUDA_NVCC_OPTIONS} --use_fast_math")
ENDIF(CUDA_BUILD_USE_FAST_MATH)

# add include directories to pass to the nvcc command
MACRO(CUDA_INCLUDE_DIRECTORIES)
    FOREACH(CUDA_INC_DIR ${ARGN})
        SET(CUDA_NVCC_INCLUDE_DIRECTORIES ${CUDA_NVCC_INCLUDE_DIRECTORIES} -I${CUDA_INC_DIR})
    ENDFOREACH(CUDA_INC_DIR ${ARGN})
ENDMACRO(CUDA_INCLUDE_DIRECTORIES)

# add definitions to pass to the nvcc command.
MACRO(CUDA_ADD_DEFINITIONS)
    FOREACH(CUDA_DEFINITION ${ARGN})
        SET(CUDA_NVCC_DEFINITIONS ${CUDA_NVCC_DEFINITIONS} ${CUDA_DEFINITION})
    ENDFOREACH(CUDA_DEFINITION ${ARGN})
ENDMACRO(CUDA_ADD_DEFINITIONS)

# add the nvcc command to all .cu files
MACRO(CUDA_ADD_NVCC_COMMANDS)
    FOREACH(CU_FILE ${ARGV})
        GET_FILENAME_COMPONENT(CU_FILE_NAME ${CU_FILE} NAME_WE)
        GET_FILENAME_COMPONENT(CU_FILE_EXT  ${CU_FILE} EXT)
        GET_FILENAME_COMPONENT(CU_FILE_PATH ${CU_FILE} PATH)
        IF (CU_FILE_EXT STREQUAL ".cu")
            IF (WIN32)
                # CMake generates a directory where the intermediate files are stored
                SET(CUDA_NVCC_OUTPUT_FILE       "${PROJECT_BINARY_DIR}/${PROJECT_NAME}.dir/$(ConfigurationName)/${CU_FILE_NAME}.obj")
                SET(CUDA_NVCC_OUTPUT_CUBIN_FILE "${PROJECT_BINARY_DIR}/${PROJECT_NAME}.dir/$(ConfigurationName)/${CU_FILE_NAME}.cubin")
            ELSE (WIN32)
                MESSAGE(FATAL_ERROR "only Win32 Platform supported at this point")
            ENDIF (WIN32)

            IF (NOT CUDA_BUILD_EMULATION AND CUDA_BUILD_DISPLAY_COMPILE_GPU_INFO)
                ADD_CUSTOM_COMMAND(OUTPUT ${CUDA_NVCC_OUTPUT_CUBIN_FILE} ${CUDA_NVCC_OUTPUT_FILE}
                                   # add the command to generate the cubin output file
                                   COMMAND  ${CUDE_NVCC_COMMAND}
                                       ARGS ${CUDA_NVCC_OPTIONS}
                                            ${CUDA_NVCC_DEFINITIONS}
                                            ${CUDA_NVCC_INCLUDE_DIRECTORIES}
                                            -cubin
                                            -o \"${CUDA_NVCC_OUTPUT_CUBIN_FILE}\"
                                            \"${CU_FILE}\"
                                   # add the command to parse the generated cubin file
                                   COMMAND  ${CMAKE_COMMAND}
                                       ARGS -D input_file="${CUDA_NVCC_OUTPUT_CUBIN_FILE}"
                                            -P "${CMAKE_SOURCE_DIR}//cmake_custom_scripts/cuda_parse_cubin.cmake"
                                   MAIN_DEPENDENCY ${CU_FILE}
                                   DEPENDS  ${CU_FILE}
                                   COMMENT "NVCC compiling (${CU_FILE}):")
            ENDIF (NOT CUDA_BUILD_EMULATION AND CUDA_BUILD_DISPLAY_COMPILE_GPU_INFO)

            # add the actual nvcc command to generate the .obj/.o file
            ADD_CUSTOM_COMMAND(OUTPUT   ${CUDA_NVCC_OUTPUT_FILE}
                               COMMAND  ${CUDE_NVCC_COMMAND}
                                   ARGS ${CUDA_NVCC_OPTIONS}
                                        ${CUDA_NVCC_DEFINITIONS}
                                        ${CUDA_NVCC_INCLUDE_DIRECTORIES}
                                        -c
                                        -o \"${CUDA_NVCC_OUTPUT_FILE}\"
                                        \"${CU_FILE}\"
                               MAIN_DEPENDENCY ${CU_FILE}
                               DEPENDS  ${CU_FILE}
                               COMMENT "NVCC compiling (${CU_FILE}):"
                               APPEND)
        ENDIF (CU_FILE_EXT STREQUAL ".cu")
    ENDFOREACH(CU_FILE ${ARGV})
ENDMACRO(CUDA_ADD_NVCC_COMMANDS)
