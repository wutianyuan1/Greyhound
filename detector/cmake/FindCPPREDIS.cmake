# FindCppRedis
# --------
#
# Find cpp_redis
#
# Find the native cpp_redis includes and library This module defines
#
# input variable:
#   CPPREDIS_MT_DIR, OPTIONAL,set path of cppredis libray built with /MT using MSVC 
#   CPPREDIS_MD_DIR, OPTIONAL,set path of cppredis libray built with /MD using MSVC
#   CPPREDIS_MSVCRT, OPTIONAL,set '/MT' or '/MD' for define type of link msvc runtime library of current cppredis library,default '/MT'
# 							 if CPPREDIS_MT_DIR or CPPREDIS_MD_DIR defined, ignore it.
# output variable:
#
#   CPPREDIS_FOUND, If false, do not try to use cpp_redis.
#   CPPREDIS_INCLUDE_DIR, where to find header cpp_redis, etc.
#   CPPREDIS_LIBRARY, where to find the cpp_redis library.
#   TACOPIE_LIBRARY,  where to find the tacopie library.
# for msvc :
#   CPPREDIS_LIBRARY_DEBUG, where to find the cpp_redis library (Debug FOR MSVC).
#   CPPREDIS_LIBRARY_RELEASE, where to find the cpp_redis library (Release FOR MSVC).
#   TACOPIE_LIBRARY_DEBUG, where to find the tacopie library.(Debug FOR MSVC).
#   TACOPIE_LIBRARY_RELEASE, where to find the tacopie library (Release FOR MSVC).
# import target:
# 	cppredis
# 	cppredis_mt (FOR MSVC)

function(msvc_create_target _msvcrt_suffix _path)
	unset(CPPREDIS_INCLUDE_DIR CACHE)
	unset(CPPREDIS_LIBRARY_DEBUG CACHE)
	unset(CPPREDIS_LIBRARY_RELEASE CACHE)
	unset(TACOPIE_LIBRARY_DEBUG CACHE)
	unset(TACOPIE_LIBRARY_RELEASE CACHE)

	if(_path)
		if(NOT EXISTS ${_path})
			message(WARNING "invalid cppredis library path ${_path}")
			return()
		endif()
		# find header file
		find_path(CPPREDIS_INCLUDE_DIR cpp_redis/cpp_redis PATHS "${_path}/include" NO_DEFAULT_PATH)

		find_library(CPPREDIS_LIBRARY_DEBUG   NAMES cpp_redis cpp_redis${_msvcrt_suffix}_d PATHS "${_path}/lib" "${_path}/lib/Debug"   NO_DEFAULT_PATH )
		find_library(CPPREDIS_LIBRARY_RELEASE NAMES cpp_redis cpp_redis${_msvcrt_suffix}   PATHS "${_path}/lib" "${_path}/lib/Release" NO_DEFAULT_PATH )
		find_library(TACOPIE_LIBRARY_DEBUG    NAMES tacopie   tacopie${_msvcrt_suffix}_d   PATHS "${_path}/lib" "${_path}/lib/Debug"   NO_DEFAULT_PATH )
		find_library(TACOPIE_LIBRARY_RELEASE  NAMES tacopie   tacopie${_msvcrt_suffix}     PATHS "${_path}/lib" "${_path}/lib/Release" NO_DEFAULT_PATH )
	else()
		# find header file
		find_path(CPPREDIS_INCLUDE_DIR cpp_redis/cpp_redis)
		find_library(CPPREDIS_LIBRARY_DEBUG   NAMES cpp_redis cpp_redis${_msvcrt_suffix}_d PATH_SUFFIXES Debug   )
		find_library(CPPREDIS_LIBRARY_RELEASE NAMES cpp_redis cpp_redis${_msvcrt_suffix}   PATH_SUFFIXES Release )
		find_library(TACOPIE_LIBRARY_DEBUG    NAMES tacopie   tacopie${_msvcrt_suffix}_d   PATH_SUFFIXES Debug   )
		find_library(TACOPIE_LIBRARY_RELEASE  NAMES tacopie   tacopie${_msvcrt_suffix}     PATH_SUFFIXES Release )
	endif()

	FIND_PACKAGE_HANDLE_STANDARD_ARGS(CPPREDIS DEFAULT_MSG 
			CPPREDIS_INCLUDE_DIR 
			CPPREDIS_LIBRARY_RELEASE 
			CPPREDIS_LIBRARY_DEBUG 
			TACOPIE_LIBRARY_RELEASE 
			TACOPIE_LIBRARY_DEBUG)
	if(${_msvcrt_suffix} STREQUAL "_mt")
		set(_target_name "cppredis_mt")
		set(_cppredis_msvcrt "/MT")
	else()
		set(_target_name "cppredis")
		set(_cppredis_msvcrt "/MD")
	endif()
	if(CPPREDIS_FOUND)
		message(STATUS "IMPORTED TARGET ${_target_name}")
		# for compatility of find_dependency
		set(CppRedis_FOUND TRUE PARENT_SCOPE)
		set(CPPREDIS_FOUND TRUE PARENT_SCOPE)
		add_library(${_target_name} STATIC IMPORTED)
		set_target_properties(${_target_name} PROPERTIES
			IMPORTED_LINK_INTERFACE_LANGUAGES CXX
			INTERFACE_INCLUDE_DIRECTORIES ${CPPREDIS_INCLUDE_DIR}
			)
		set_target_properties(${_target_name} PROPERTIES
			IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG ${TACOPIE_LIBRARY_DEBUG}
			IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE ${TACOPIE_LIBRARY_RELEASE}
			IMPORTED_LOCATION_DEBUG ${CPPREDIS_LIBRARY_DEBUG}
			IMPORTED_LOCATION_RELEASE ${CPPREDIS_LIBRARY_RELEASE}
			# set /MT or /MT option
			INTERFACE_COMPILE_OPTIONS "${_cppredis_msvcrt}$<$<CONFIG:Debug>:d>"
			)
		set(_msvcrt_flags "${_msvcrt_flags}${_cppredis_msvcrt}" PARENT_SCOPE)
	endif(CPPREDIS_FOUND)
endfunction(msvc_create_target )

# handle the QUIETLY and REQUIRED arguments and set CPPREDIS_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
if(MSVC)
	unset(_msvcrt_flags)
	if(CPPREDIS_MT_DIR)
		unset(CPPREDIS_FOUND)
		unset(CppRedis_FOUND)
		msvc_create_target( "_mt" "${CPPREDIS_MT_DIR}")
	endif()

	if(CPPREDIS_MD_DIR)
		unset(CPPREDIS_FOUND)
		unset(CppRedis_FOUND)
		msvc_create_target( "_md" "${CPPREDIS_MD_DIR}")
	endif()

	if(NOT CPPREDIS_FOUND)
		if ("${CPPREDIS_MSVCRT}" STREQUAL "")
			set(CPPREDIS_MSVCRT "/MT" PARENT_SCOPE)
		endif ()
		if(${CPPREDIS_MSVCRT} MATCHES "/MT")
			msvc_create_target( "_mt" "")
		elseif(${CPPREDIS_MSVCRT} MATCHES "/MD")
			msvc_create_target( "_md" "")
		endif()
		set(CPPREDIS_LIBRARY ${CPPREDIS_LIBRARY_RELEASE})
		set(TACOPIE_LIBRARY  ${TACOPIE_LIBRARY_RELEASE} )
	else()
		set(CPPREDIS_MSVCRT ${_msvcrt_flags})
	endif(NOT CPPREDIS_FOUND)
	mark_as_advanced(CPPREDIS_LIBRARY_DEBUG  CPPREDIS_LIBRARY_RELEASE TACOPIE_LIBRARY_DEBUG TACOPIE_LIBRARY_RELEASE CPPREDIS_INCLUDE_DIR )
else()
	# find header file
	find_path(CPPREDIS_INCLUDE_DIR cpp_redis/cpp_redis)

	find_library(CPPREDIS_LIBRARY cpp_redis )
	find_library(TACOPIE_LIBRARY tacopie )
	FIND_PACKAGE_HANDLE_STANDARD_ARGS(CPPREDIS DEFAULT_MSG CPPREDIS_LIBRARY TACOPIE_LIBRARY CPPREDIS_INCLUDE_DIR)
	if(CPPREDIS_FOUND)
	# for compatility of find_dependency
	set (CppRedis_FOUND TRUE)
	add_library(cppredis STATIC IMPORTED)
	set_target_properties(cppredis PROPERTIES
		IMPORTED_LINK_INTERFACE_LANGUAGES CXX
		INTERFACE_INCLUDE_DIRECTORIES ${CPPREDIS_INCLUDE_DIR}
		)

	set_target_properties(cppredis PROPERTIES
		IMPORTED_LINK_INTERFACE_LIBRARIES "${TACOPIE_LIBRARY};ws2_32"
		IMPORTED_LOCATION "${CPPREDIS_LIBRARY}"
		)
	mark_as_advanced(CPPREDIS_LIBRARY TACOPIE_LIBRARY CPPREDIS_INCLUDE_DIR )
endif()
endif(MSVC)