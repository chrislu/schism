
# Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
# Distributed under the Modified BSD License, see license.txt.

# SCM_CGC_FRAGMENT_PROFILE
# G80           gp4fp
# G70, NV40     fp40

# SCM_CGC_VERTEX_PROFILE
# G80           gp4vp
# G70, NV40     vp40

# SCM_CGC_GEOMETRY_PROFILE
# G80           gp4gp

SET(SCM_CGC_MIN_VERSION_MAJOR 2)
SET(SCM_CGC_MIN_VERSION_MINOR 0)
SET(SCM_CGC_MIN_VERSION_BUILD 0)
SET(SCM_CGC_MIN_VERSION "${SCM_CGC_MIN_VERSION_MAJOR}.${SCM_CGC_MIN_VERSION_MINOR}.${SCM_CGC_MIN_VERSION_BUILD}")

IF (NOT SCM_CGC_EXECUTABLE)

    FIND_PROGRAM(SCM_CGC_EXECUTABLE "cgc" PATHS ${GLOBAL_EXT_DIR}/bin/cg
                                                /opt/nvidia/cg/latest/usr/bin
                                                $ENV{CG_BIN_PATH}
                                                $ENV{CG_BIN64_PATH}
                                                $ENV{PATH} NO_DEFAULT_PATH)

    IF (NOT SCM_CGC_EXECUTABLE)
        MESSAGE(FATAL_ERROR "unable to find cgc compiler")
    ENDIF (NOT SCM_CGC_EXECUTABLE)

    EXEC_PROGRAM(${SCM_CGC_EXECUTABLE} ARGS "-v" OUTPUT_VARIABLE TMP_CGC_OUTPUT)

    STRING(REGEX REPLACE ".*version ([0-9]+\\.[0-9]+\\.[0-9]+).*" "\\1" SCM_CGC_VERSION ${TMP_CGC_OUTPUT})

    STRING(REGEX REPLACE "([0-9]+)\\.[0-9]+\\.[0-9]+" "\\1" SCM_CGC_VERSION_MAJOR ${SCM_CGC_VERSION})
    STRING(REGEX REPLACE "[0-9]+\\.([0-9]+)\\.[0-9]+" "\\1" SCM_CGC_VERSION_MINOR ${SCM_CGC_VERSION})
    STRING(REGEX REPLACE "[0-9]+\\.[0-9]+\\.([0-9]+)" "\\1" SCM_CGC_VERSION_BUILD ${SCM_CGC_VERSION})

    MATH(EXPR SCM_CGC_VERSION_NUM     "${SCM_CGC_VERSION_MAJOR}*10000     + ${SCM_CGC_VERSION_MINOR}*100     + ${SCM_CGC_VERSION_BUILD}")
    MATH(EXPR SCM_CGC_MIN_VERSION_NUM "${SCM_CGC_MIN_VERSION_MAJOR}*10000 + ${SCM_CGC_MIN_VERSION_MINOR}*100 + ${SCM_CGC_MIN_VERSION_BUILD}")

    IF (SCM_CGC_VERSION_NUM LESS SCM_CGC_MIN_VERSION_NUM)
        MESSAGE(FATAL_ERROR "found cgc version ${SCM_CGC_VERSION} to old (min. version ${SCM_CGC_MIN_VERSION} required)")
    ENDIF (SCM_CGC_VERSION_NUM LESS SCM_CGC_MIN_VERSION_NUM)

ENDIF( NOT SCM_CGC_EXECUTABLE)

macro(scm_add_cgc_glsl_command SCM_CGC_OUTPUT_FILES)
    set(input_file_list  ${ARGV})
    set(output_file_list )
    set(output_file_cgc_profile "")
    # remove the leading output variable
    list(REMOVE_AT input_file_list 0)

    foreach(input_file ${input_file_list})
        get_filename_component(input_file           ${input_file} ABSOLUTE)
        get_filename_component(input_file_name      ${input_file} NAME)
        get_filename_component(input_file_name_we   ${input_file} NAME_WE)
        get_filename_component(input_file_ext       ${input_file} EXT)
        get_filename_component(input_file_path      ${input_file} PATH)
        get_filename_component(project_src_path     ${SCM_PROJECT_SOURCE_DIR} ABSOLUTE)

        if (input_file_ext STREQUAL ".glslf")
            if (NOT SCM_CGC_FRAGMENT_PROFILE)
                MESSAGE(FATAL_ERROR "no fragment profile defined. define SCM_CGC_FRAGMENT_PROFILE (see /build/cmake/schism_glsl.cmake)")
            endif (NOT SCM_CGC_FRAGMENT_PROFILE)
            set(output_file_cgc_profile ${SCM_CGC_FRAGMENT_PROFILE})
        elseif (input_file_ext STREQUAL ".glslv")
            if (NOT SCM_CGC_VERTEX_PROFILE)
                MESSAGE(FATAL_ERROR "no fragment profile defined. define SCM_CGC_VERTEX_PROFILE (see /build/cmake/schism_glsl.cmake)")
            endif (NOT SCM_CGC_VERTEX_PROFILE)
            set(output_file_cgc_profile ${SCM_CGC_VERTEX_PROFILE})
        elseif (input_file_ext STREQUAL ".glslg")
            if (NOT SCM_CGC_GEOMETRY_PROFILE)
                MESSAGE(FATAL_ERROR "no fragment profile defined. define SCM_CGC_GEOMETRY_PROFILE (see /build/cmake/schism_glsl.cmake)")
            endif (NOT SCM_CGC_GEOMETRY_PROFILE)
            set(output_file_cgc_profile ${SCM_CGC_GEOMETRY_PROFILE})
        else (input_file_ext STREQUAL ".glslf")
            message("the found shader file extension (${input_file_ext}) is not supported by this script, so no offline compile support will be generated for this file (${input_file})")
        endif (input_file_ext STREQUAL ".glslf")

        if (NOT output_file_cgc_profile STREQUAL "")
            set(output_file_path "${input_file_path}/_cgc_glsl_generated")
            file(MAKE_DIRECTORY ${output_file_path})
            set(output_file  "${output_file_path}/${input_file_name}.arb_asm")
            list(APPEND output_file_list ${output_file})

            file(RELATIVE_PATH project_path ${project_src_path} ${output_file_path})
            if (project_path)
                file(TO_CMAKE_PATH ${project_path} project_path)
            endif (project_path)

            source_group(source_files/${project_path} FILES ${output_file})

                                        #-glslWerror

            add_custom_command(OUTPUT   ${output_file}
                               COMMAND  ${SCM_CGC_EXECUTABLE}
                                   ARGS -o \"${output_file}\"
                                        -oglsl
                                        -strict
                                        -quiet
                                        -profile ${output_file_cgc_profile}
                                        \"${input_file}\"
                               MAIN_DEPENDENCY ${input_file}
                               DEPENDS  ${input_file}
                               COMMENT "cgc ${input_file_name}")
        endif (NOT output_file_cgc_profile STREQUAL "")

    endforeach(input_file)

    set_source_files_properties(${output_file_list} PROPERTIES GENERATED TRUE)
    list(APPEND ${SCM_CGC_OUTPUT_FILES} ${output_file_list})

    #set(${SCM_CGC_OUTPUT_FILES} ${output_file_list})
    #set_source_files_properties(${SCM_CGC_OUTPUT_FILES} PROPERTIES GENERATED TRUE)

endmacro(scm_add_cgc_glsl_command SCM_CGC_OUTPUT_FILES)
