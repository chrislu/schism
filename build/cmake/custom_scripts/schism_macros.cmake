
# Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
# Distributed under the Modified BSD License, see license.txt.

#include(GetPrerequisites)

macro(scm_project_files)
    set(out_file_list     ${ARGV0})
    set(in_project_path   ${ARGV1})
    set(in_glob_expr      ${ARGV})

    get_filename_component(project_src_path     ${SCM_PROJECT_SOURCE_DIR} ABSOLUTE)
    get_filename_component(in_project_src_path  ${in_project_path} ABSOLUTE)

    #message(${SCM_PROJECT_SOURCE_DIR})
    #message(${project_src_path})
    #message(${in_project_src_path})

    # remove the leading output and input variable
    list(REMOVE_AT in_glob_expr 0 1)
    set(out_proj_files    "")

    set(SOURCE_GROUP_DELIMITER "/")
    file(RELATIVE_PATH project_path ${project_src_path} ${in_project_src_path})
    if (project_path)
        file(TO_CMAKE_PATH ${project_path} project_path)
    endif (project_path)

    foreach(glob_expr ${in_glob_expr})
        file(GLOB proj_files ${in_project_src_path}/${glob_expr})
        foreach(proj_file ${proj_files})
            list(APPEND out_proj_files ${proj_file})
            #message(${project_path}  ${proj_file})
            source_group(source_files/${project_path} FILES ${proj_file})
        endforeach(proj_file)
    endforeach(glob_expr)

    list(APPEND ${out_file_list} ${out_proj_files})
endmacro(scm_project_files)

macro(scm_project_include_directories)
    set(in_platform         ${ARGV0})
    set(in_directories      ${ARGV})

	list(REMOVE_AT in_directories 0)
	set(DIR_LIST_NAME "${PROJECT_NAME}_INCLUDE_DIRS")

    if (${in_platform} MATCHES ALL)
		list(APPEND ${DIR_LIST_NAME} ${in_directories})
		include_directories(${in_directories})
	else (${in_platform} MATCHES ALL)
		if (${in_platform})
            list(APPEND ${DIR_LIST_NAME} ${in_directories})
			include_directories(${in_directories})
		endif (${in_platform})
    endif (${in_platform} MATCHES ALL)

	#message(${PROJECT_NAME} ${${DIR_LIST_NAME}})
endmacro(scm_project_include_directories)

macro(scm_project_link_directories)
    set(in_platform         ${ARGV0})
    set(in_directories      ${ARGV})

	list(REMOVE_AT in_directories 0)

    if (in_platform MATCHES ALL)
		link_directories(${in_directories})
	else (in_platform MATCHES ALL)
		if (${in_platform})
			link_directories(${in_directories})
		endif (${in_platform})
    endif (in_platform MATCHES ALL)
endmacro(scm_project_link_directories)

macro(scm_link_libraries)
    set(in_platform         ${ARGV0})
    set(in_libraries        ${ARGV})

	list(REMOVE_AT in_libraries 0)

	# first to the actual target_link_libraries
    if (in_platform MATCHES ALL)
		target_link_libraries(${PROJECT_NAME} ${in_libraries})
	else (in_platform MATCHES ALL)
		if (${in_platform})
			target_link_libraries(${PROJECT_NAME} ${in_libraries})
		endif (${in_platform})
    endif (in_platform MATCHES ALL)

	get_target_property(target_type ${PROJECT_NAME} TYPE)
	if (target_type MATCHES EXECUTABLE)
		if (in_platform MATCHES ALL OR in_platform MATCHES WIN32)
			get_directory_property(link_dirs LINK_DIRECTORIES)
			list(REMOVE_ITEM in_libraries debug optimized general)
			list(REMOVE_ITEM link_dirs ${SCHISM_LIBRARY_DIR})
			set(conf_list release debug)

			foreach(conf ${conf_list})
				foreach(link_lib ${in_libraries})
					#message(${link_lib})
					if (link_lib STREQUAL "cuda")
						set(link_lib ${SCM_CUDA_SHARED_LIB_NAME})
						#message(${link_lib})
					endif (link_lib STREQUAL "cuda")
					set(link_lib_dll ${link_lib}.dll)
					foreach(ldir ${link_dirs})
						set(dll_path ${ldir}/${conf}/${link_lib_dll})
						if (EXISTS ${dll_path})
							# make if fail for the wrong configuration
							string(REPLACE /${conf}/ "/$<CONFIGURATION>/" dll_path ${dll_path})
							add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
											   COMMAND if EXIST ${dll_path} ${CMAKE_COMMAND} -E echo "copying ${link_lib_dll} to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIGURATION>/"
											   COMMAND if EXIST ${dll_path} ${CMAKE_COMMAND} -E copy_if_different ${dll_path} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIGURATION>/)
						endif (EXISTS ${dll_path})
					endforeach(ldir)
				endforeach(link_lib)
			endforeach(conf)
		endif (in_platform MATCHES ALL OR in_platform MATCHES WIN32)
	endif (target_type MATCHES EXECUTABLE)
endmacro(scm_link_libraries)

macro(scm_copy_shared_libraries)
    set(in_platform         ${ARGV0})
    set(in_libraries        ${ARGV})

	list(REMOVE_AT in_libraries 0)
	#message("in " ${in_libraries})

    if (${in_platform} OR in_platform MATCHES ALL)
		foreach(shlib ${in_libraries})
			#message("shlib " ${shlib})
			#message(${in_platform})
			#file(GLOB_RECURSE scm_glob_dlls ${GLOBAL_EXT_DIR}/lib/$<CONFIGURATION>/${shlib}.dll)
			#message(${scm_glob_dlls})
			set(scm_copy_dlls ${GLOBAL_EXT_DIR}/lib/$<CONFIGURATION>/${shlib}.dll)
			#message("scmcp " ${scm_copy_dlls})
			set(scm_copy_path ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIGURATION>/)
			string(REPLACE "/" "\\" scm_copy_dlls ${scm_copy_dlls})
			string(REPLACE "/" "\\" scm_copy_path ${scm_copy_path})
			
			#message{${scm_copy_dlls})
			#message(${scm_copy_path})
			add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
							   COMMAND if EXIST ${scm_copy_dlls} ${CMAKE_COMMAND} -E echo "copying ${shlib} to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIGURATION>/"
							   COMMAND if EXIST ${scm_copy_dlls} ${CMAKE_COMMAND} -E copy_if_different ${scm_copy_dlls} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIGURATION>/)

		endforeach(shlib)

	#	set(scm_copy_dlls ${SCHISM_LIBRARY_DIR}/$<CONFIGURATION>/*.dll)
	#	set(scm_copy_path ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIGURATION>/)
	#	string(REPLACE "/" "\\" scm_copy_dlls ${scm_copy_dlls})
	#	string(REPLACE "/" "\\" scm_copy_path ${scm_copy_path})
#
#		add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
#						   COMMAND if EXIST ${dll_path} ${CMAKE_COMMAND} -E echo "copying ${link_lib_dll} to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIGURATION>/"
#						   COMMAND if EXIST ${dll_path} ${CMAKE_COMMAND} -E copy_if_different ${dll_path} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIGURATION>/)

    endif (${in_platform} OR in_platform MATCHES ALL)

endmacro(scm_copy_shared_libraries)

macro(scm_copy_schism_libraries)
    scm_copy_shared_libraries(WIN32 ${SCM_CUDA_SHARED_LIB_NAME})
	if (NOT SCHISM_BUILD_STATIC AND WIN32)
		set(scm_copy_dlls ${SCHISM_LIBRARY_DIR}/$<CONFIGURATION>/*.dll)
		set(scm_copy_path ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIGURATION>/)
		string(REPLACE "/" "\\" scm_copy_dlls ${scm_copy_dlls})
		string(REPLACE "/" "\\" scm_copy_path ${scm_copy_path})
		add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
						   COMMAND ${CMAKE_COMMAND} -E echo "copying schism libraries to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIGURATION>/"
						   COMMAND xcopy ${scm_copy_dlls} ${scm_copy_path} /Y /C)
	endif (NOT SCHISM_BUILD_STATIC AND WIN32)
endmacro(scm_copy_schism_libraries)
