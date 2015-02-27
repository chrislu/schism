
# Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
# Distributed under the Modified BSD License, see license.txt.

include(schism_platform)

#option(SCHISM_CUDA_BUILD_DEBUG                     "Build debug"                            OFF)
option(SCHISM_CUDA_BUILD_VERBOSE                   "Verbose Output"                         OFF)
option(SCHISM_CUDA_BUILD_COMPUTE20                 "Enable Compute 2.0"                     ON)
option(SCHISM_CUDA_BUILD_COMPUTE30                 "Enable Compute 3.0"                     OFF)
option(SCHISM_CUDA_BUILD_COMPUTE35                 "Enable Compute 3.5"                     OFF)
option(SCHISM_CUDA_BUILD_COMPUTE50                 "Enable Compute 5.0"                     OFF)
option(SCHISM_CUDA_BUILD_COMPUTE52                 "Enable Compute 5.2"                     OFF)
option(SCHISM_CUDA_BUILD_USE_FAST_MATH             "Use lower precision math calculations"  ON)
option(SCHISM_CUDA_BUILD_KEEP_INTERMEDIATE_FILES   "Keep the generated intermediate files"  ON)

set(SCM_CUDA_NVCC_OPTIONS "")

if (WIN32)
    if (MSVC10)
        set (SCM_CUDA_NVCC_OPTIONS --cl-version 2010 CACHE STRING "schism cuda internal" FORCE)
        set(SCM_CUDA_SHARED_LIB_NAME "cudart64_50_32" CACHE STRING "schism cuda internal" FORCE)
    endif(MSVC10)

    if (MSVC11)
        set (SCM_CUDA_NVCC_OPTIONS --cl-version 2012 CACHE STRING "schism cuda internal" FORCE)
        set(SCM_CUDA_SHARED_LIB_NAME "cudart64_55" CACHE STRING "schism cuda internal" FORCE)
    endif(MSVC11)
	
    if (MSVC12)
        set (SCM_CUDA_NVCC_OPTIONS --cl-version 2013 CACHE STRING "schism cuda internal" FORCE)
        set(SCM_CUDA_SHARED_LIB_NAME "cudart64_65" CACHE STRING "schism cuda internal" FORCE)
    endif(MSVC12)

    set(SCM_CUDA_NVCC_OPTIONS ${SCM_CUDA_NVCC_OPTIONS} --use-local-env --compile CACHE STRING "schism cuda internal" FORCE)
    set(SCM_CUDA_NVCC_PATH                  ${GLOBAL_EXT_DIR}/bin/cuda/bin       CACHE PATH   "schism cuda internal")
    if (${SCHISM_PLATFORM} MATCHES ${PLATFORM_WIN64})
        set(SCM_CUDA_NVCC_OPTIONS  ${SCM_CUDA_NVCC_OPTIONS} --machine 64 -ccbin \"$(VCInstallDir)bin/x86_amd64/cl.exe\"  CACHE STRING "schism cuda internal" FORCE)
	else (${SCHISM_PLATFORM} MATCHES ${PLATFORM_WIN64})
        set(SCM_CUDA_NVCC_OPTIONS ${SCM_CUDA_NVCC_OPTIONS}  --machine 32 -ccbin \"$(VCInstallDir)bin/x86_amd64/cl.exe\"  CACHE STRING "schism cuda internal" FORCE)
	endif (${SCHISM_PLATFORM} MATCHES ${PLATFORM_WIN64})
elseif (UNIX)
    if (SCM_EXT_CUDA_BIN_DIR)
        set(SCM_CUDA_NVCC_PATH                  ${SCM_EXT_CUDA_BIN_DIR}                   CACHE PATH   "schism cuda internal" FORCE)
    else (SCM_EXT_CUDA_BIN_DIR)
        set(SCM_CUDA_NVCC_PATH                  /opt/cuda/current/cuda/bin                CACHE PATH   "schism cuda internal" FORCE)
    endif (SCM_EXT_CUDA_BIN_DIR)
    set(SCM_CUDA_NVCC_OPTIONS ${SCM_CUDA_NVCC_OPTIONS} --compile --machine 64 CACHE STRING "schism cuda internal" FORCE)
endif (WIN32)

set(SCM_CUDA_NVCC_COMMAND "${SCM_CUDA_NVCC_PATH}/nvcc" CACHE STRING "schism cuda internal")

# convert CXX compiler options into comma separated list
string(REPLACE " " "," SCM_CUDA_NVCC_CXX_FLAGS          ${CMAKE_CXX_FLAGS})
string(REPLACE " " "," SCM_CUDA_NVCC_CXX_FLAGS_RELEASE  ${CMAKE_CXX_FLAGS_RELEASE})
string(REPLACE " " "," SCM_CUDA_NVCC_CXX_FLAGS_DEBUG    ${CMAKE_CXX_FLAGS_DEBUG})

string(REPLACE ",-std=c++0x" "," SCM_CUDA_NVCC_CXX_FLAGS          ${SCM_CUDA_NVCC_CXX_FLAGS})
string(REPLACE ",-std=c++0x" "," SCM_CUDA_NVCC_CXX_FLAGS_RELEASE  ${SCM_CUDA_NVCC_CXX_FLAGS_RELEASE})
string(REPLACE ",-std=c++0x" "," SCM_CUDA_NVCC_CXX_FLAGS_DEBUG    ${SCM_CUDA_NVCC_CXX_FLAGS_DEBUG})

# eat multiple commas
string(REGEX REPLACE "[\\,]+" "," SCM_CUDA_NVCC_CXX_FLAGS          ${SCM_CUDA_NVCC_CXX_FLAGS})
string(REGEX REPLACE "[\\,]+" "," SCM_CUDA_NVCC_CXX_FLAGS_RELEASE  ${SCM_CUDA_NVCC_CXX_FLAGS_RELEASE})
string(REGEX REPLACE "[\\,]+" "," SCM_CUDA_NVCC_CXX_FLAGS_DEBUG    ${SCM_CUDA_NVCC_CXX_FLAGS_DEBUG})

set(SCM_CUDA_NVCC_DEFINITIONS           "")
set(SCM_CUDA_NVCC_INCLUDE_DIRECTORIES   "")

#if (SCHISM_CUDA_BUILD_DEBUG)
#    set(SCM_CUDA_NVCC_OPTIONS ${SCM_CUDA_NVCC_OPTIONS} -Xcompiler ${SCM_CUDA_NVCC_CXX_FLAGS_DEBUG}${SCM_CUDA_NVCC_CXX_FLAGS} CACHE STRING "schism cuda internal" FORCE)
#else (SCHISM_CUDA_BUILD_DEBUG)
#    set(SCM_CUDA_NVCC_OPTIONS ${SCM_CUDA_NVCC_OPTIONS} --optimize 3 -Xcompiler ${SCM_CUDA_NVCC_CXX_FLAGS_RELEASE}${SCM_CUDA_NVCC_CXX_FLAGS} CACHE STRING "schism cuda internal" FORCE)
#endif (SCHISM_CUDA_BUILD_DEBUG)

set(SCM_CUDA_NVCC_OPTIONS ${SCM_CUDA_NVCC_OPTIONS} -Xcompiler                                                                           CACHE STRING "schism cuda internal" FORCE)
set(SCM_CUDA_NVCC_OPTIONS ${SCM_CUDA_NVCC_OPTIONS} $<$<CONFIG:Debug>:${SCM_CUDA_NVCC_CXX_FLAGS_DEBUG}${SCM_CUDA_NVCC_CXX_FLAGS}>        CACHE STRING "schism cuda internal" FORCE)
set(SCM_CUDA_NVCC_OPTIONS ${SCM_CUDA_NVCC_OPTIONS} $<$<CONFIG:Release>:${SCM_CUDA_NVCC_CXX_FLAGS_RELEASE}${SCM_CUDA_NVCC_CXX_FLAGS}>    CACHE STRING "schism cuda internal" FORCE)
set(SCM_CUDA_NVCC_OPTIONS ${SCM_CUDA_NVCC_OPTIONS} $<$<CONFIG:Release>:--optimize> $<$<CONFIG:Release>:3>                               CACHE STRING "schism cuda internal" FORCE)

if (SCHISM_CUDA_BUILD_VERBOSE)
  set(SCM_CUDA_NVCC_OPTIONS ${SCM_CUDA_NVCC_OPTIONS} --ptxas-options=-v CACHE STRING "schism cuda internal" FORCE)
endif (SCHISM_CUDA_BUILD_VERBOSE)

if (SCHISM_CUDA_BUILD_COMPUTE20)
  set(SCM_CUDA_NVCC_OPTIONS ${SCM_CUDA_NVCC_OPTIONS} -gencode arch=\"compute_20\",code=\"sm_20\" CACHE STRING "schism cuda internal" FORCE)
endif (SCHISM_CUDA_BUILD_COMPUTE20)

if (SCHISM_CUDA_BUILD_COMPUTE30)
  set(SCM_CUDA_NVCC_OPTIONS ${SCM_CUDA_NVCC_OPTIONS} -gencode arch=\"compute_30\",code=\"sm_30\" CACHE STRING "schism cuda internal" FORCE)
endif (SCHISM_CUDA_BUILD_COMPUTE30)

if (SCHISM_CUDA_BUILD_COMPUTE35)
  set(SCM_CUDA_NVCC_OPTIONS ${SCM_CUDA_NVCC_OPTIONS} -gencode arch=\"compute_35\",code=\"sm_35\" CACHE STRING "schism cuda internal" FORCE)
endif (SCHISM_CUDA_BUILD_COMPUTE35)

if (SCHISM_CUDA_BUILD_COMPUTE50)
  set(SCM_CUDA_NVCC_OPTIONS ${SCM_CUDA_NVCC_OPTIONS} -gencode arch=\"compute_50\",code=\"sm_50\" CACHE STRING "schism cuda internal" FORCE)
endif (SCHISM_CUDA_BUILD_COMPUTE50)

if (SCHISM_CUDA_BUILD_COMPUTE52)
  set(SCM_CUDA_NVCC_OPTIONS ${SCM_CUDA_NVCC_OPTIONS} -gencode arch=\"compute_52\",code=\"sm_52\" CACHE STRING "schism cuda internal" FORCE)
endif (SCHISM_CUDA_BUILD_COMPUTE52)

if (SCHISM_CUDA_BUILD_KEEP_INTERMEDIATE_FILES)
    if (WIN32)
        set(SCM_CUDA_NVCC_OPTIONS "${SCM_CUDA_NVCC_OPTIONS} --keep-dir \"x64\\Release\"" CACHE STRING "schism cuda internal" FORCE)
    endif (WIN32)
endif(SCHISM_CUDA_BUILD_KEEP_INTERMEDIATE_FILES)

set(SCM_CUDA_NVCC_OPTIONS ${SCM_CUDA_NVCC_OPTIONS} -restrict -use_fast_math -fmad=true -ftz=true -prec-div=false -prec-sqrt=false -Xptxas -v,-dlcm=ca CACHE STRING "schism cuda internal" FORCE)
#if (SCHISM_CUDA_BUILD_USE_FAST_MATH)
#    set(SCM_CUDA_NVCC_OPTIONS ${SCM_CUDA_NVCC_OPTIONS} --use_fast_math --fmad=true  CACHE STRING "schism cuda internal" FORCE)
#endif(SCHISM_CUDA_BUILD_USE_FAST_MATH)

# add include directories to pass to the nvcc command
set(SCM_CUDA_NVCC_INCLUDE_DIRECTORIES "")
macro(scm_cuda_project_include_directories)
	list(APPEND SCM_CUDA_NVCC_INCLUDE_DIRECTORIES ${ARGN})
endmacro(scm_cuda_project_include_directories)

# add definitions to pass to the nvcc command.
set(SCM_CUDA_NVCC_DEFINITIONS "")
macro(scm_cuda_definitions)
	list(APPEND SCM_CUDA_NVCC_DEFINITIONS ${ARGN})
endmacro(scm_cuda_definitions)

# add the nvcc command to all .cu files
macro(scm_cuda_add_nvcc_prepass SCM_CUDA_OUTPUT_FILES)
    set(input_file_list  ${ARGV})
    set(output_file_list )

    list(REMOVE_AT input_file_list 0)

	set(inc_dir_list_name "${PROJECT_NAME}_INCLUDE_DIRS")
	set(SCM_CUDA_NVCC_INCLUDE_DIRECTORIES ${${inc_dir_list_name}} ${SCM_CUDA_NVCC_INCLUDE_DIRECTORIES})

	foreach(cuda_idir ${SCM_CUDA_NVCC_INCLUDE_DIRECTORIES})
		set(SCM_CUDA_NVCC_INC_DIR_STRING ${SCM_CUDA_NVCC_INC_DIR_STRING} -I${cuda_idir})
    endforeach(cuda_idir ${SCM_CUDA_NVCC_INCLUDE_DIRECTORIES})

	foreach(cuda_def ${SCM_CUDA_NVCC_DEFINITIONS})
		set(SCM_CUDA_NVCC_DEF_STRING ${SCM_CUDA_NVCC_DEF_STRING} -I${cuda_def})
    endforeach(cuda_def ${SCM_CUDA_NVCC_DEFINITIONS})

    # remove the leading output variable
    #LIST(REMOVE_AT input_file_list 0)

    foreach(input_file ${input_file_list})
        get_filename_component(input_file           ${input_file} ABSOLUTE)
        get_filename_component(input_file_name      ${input_file} NAME)
        get_filename_component(input_file_name_we   ${input_file} NAME_WE)
        get_filename_component(input_file_ext       ${input_file} EXT)
        get_filename_component(input_file_path      ${input_file} PATH)

        if (input_file_ext STREQUAL ".cu")

            if (WIN32)
                # CMake generates a directory where the intermediate files are stored
                file(MAKE_DIRECTORY             "${PROJECT_BINARY_DIR}/${PROJECT_NAME}.dir/$(ConfigurationName)")
                set(CUDA_NVCC_OUTPUT_FILE       ${PROJECT_BINARY_DIR}/${PROJECT_NAME}.dir/$(ConfigurationName)/${input_file_name_we}.obj)
                set(CUDA_NVCC_OUTPUT_CUBIN_FILE ${PROJECT_BINARY_DIR}/${PROJECT_NAME}.dir/$(ConfigurationName)/${input_file_name_we}.cubin)
            endif (WIN32)
            if (UNIX)
                # CMake generates a directory where the intermediate files are stored
                file(MAKE_DIRECTORY             "${PROJECT_BINARY_DIR}/CMakeFiles/${PROJECT_NAME}.dir${input_file_path}")
                set(CUDA_NVCC_OUTPUT_FILE       "${PROJECT_BINARY_DIR}/CMakeFiles/${PROJECT_NAME}.dir${input_file_path}/${input_file_name_we}.o")
                set(CUDA_NVCC_OUTPUT_CUBIN_FILE "${PROJECT_BINARY_DIR}/CMakeFiles/${PROJECT_NAME}.dir${input_file_path}/${input_file_name_we}.cubin")
            endif (UNIX)

            set(output_file_list ${output_file_list} ${CUDA_NVCC_OUTPUT_FILE})

            # add the actual nvcc command to generate the .obj/.o file
            add_custom_command(OUTPUT   ${CUDA_NVCC_OUTPUT_FILE}
                               COMMAND  ${SCM_CUDA_NVCC_COMMAND}
                                   ARGS ${SCM_CUDA_NVCC_OPTIONS}
                                        ${SCM_CUDA_NVCC_INC_DIR_STRING}
                                        ${SCM_CUDA_NVCC_DEF_STRING}
                                        -o \"${CUDA_NVCC_OUTPUT_FILE}\"
                                        \"${input_file}\"
                               MAIN_DEPENDENCY ${input_file}
                               DEPENDS  ${input_file}
                               COMMENT "NVCC compiling (${input_file_name}):")

        endif (input_file_ext STREQUAL ".cu")
    endforeach(input_file)

    set_source_files_properties(${output_file_list} PROPERTIES GENERATED TRUE)
    set(${SCM_CUDA_OUTPUT_FILES} ${output_file_list})

endmacro(scm_cuda_add_nvcc_prepass)

