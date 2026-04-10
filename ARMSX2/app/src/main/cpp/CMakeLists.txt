# Setting it to a range tells it that it supports the features on the newer
# versions as well, avoiding setting policies.
cmake_minimum_required(VERSION 3.16...3.25)

# Enabling this cmake policy gets rid of warnings regarding LTO.
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	message(FATAL_ERROR "PCSX2 does not support in-tree builds.  Please make a build directory that is not the PCSX2 source directory and generate your CMake project there using either `cmake -B build_directory` or by running cmake from the build directory.")
endif()

# Project Name
project(Pcsx2 C CXX)

# Variable to check that people use the good file
set(TOP_CMAKE_WAS_SOURCED TRUE)

# set module path
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Write binaries to the bin directory under the build directory.
# This makes it simple to run development builds directly out of the build directory.
if (UNIX AND NOT APPLE)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
endif()

# include some generic functions to ensure correctness of the env
include(Pcsx2Utils)

check_no_parenthesis_in_path()
detect_operating_system()
detect_compiler()

# Disable desktop-only features on Android
if(ANDROID)
	set(ENABLE_QT_UI OFF CACHE BOOL "" FORCE)
endif()

#-------------------------------------------------------------------------------
# Include specific module
include(BuildParameters)
include(SearchForStuff)

# Must be done after SearchForStuff
get_git_version_info()
write_svnrev_h()

# make common
add_subdirectory(common)

# make pcsx2
add_subdirectory(pcsx2)
#add_subdirectory(pcsx2-qt)

# Android JNI shared library
if(ANDROID)
	add_subdirectory(3rdparty/googletest EXCLUDE_FROM_ALL)
	add_library(emucore SHARED
		native-lib.cpp
		tests/arm64/codegen_tests.cpp
		tests/arm64/codegen_tests_main.cpp
		tests/arm64/run_tests.cpp
		tests/core/run_patch_tests.cpp
		tests/ee/run_ee_seq_tests.cpp
		tests/ee/run_ee_tests.cpp
		tests/mvu/run_mvu_tests.cpp
		tests/vif/run_vif_tests.cpp
	)
	target_link_libraries(emucore PRIVATE
		PCSX2
		common
		imgui
		gtest
		libzip::zip
		SDL3::SDL3
		shaderc
		android
		log
	)
	target_include_directories(emucore PRIVATE
		"${CMAKE_SOURCE_DIR}"
		"${CMAKE_SOURCE_DIR}/pcsx2"
		"${CMAKE_SOURCE_DIR}/3rdparty/vixl/include"
	)
endif()

# Desktop Linux executable
if(NOT ANDROID AND UNIX AND NOT APPLE)
	add_executable(armsx2
		desktop/main.cpp
	)
	target_link_libraries(armsx2 PRIVATE
		PCSX2
		common
		imgui
		libzip::zip
		SDL3::SDL3
		simpleini
	)
	target_include_directories(armsx2 PRIVATE
		"${CMAKE_SOURCE_DIR}"
		"${CMAKE_SOURCE_DIR}/pcsx2"
		"${CMAKE_SOURCE_DIR}/3rdparty/vixl/include"
	)
	target_compile_definitions(armsx2 PRIVATE
		ARMSX2_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
	)
endif()

# Updater is Windows only for now.
if (WIN32)
	add_subdirectory(updater)
endif()

## tests
#if(ENABLE_TESTS)
#	add_subdirectory(3rdparty/googletest)
#	add_subdirectory(tests/ctest)
#endif()

# gsrunner
if(ENABLE_GSRUNNER)
	if (NOT WIN32)
		message(WARNING "GSRunner is only supported on Windows and may not build on your system")
	endif()
	add_subdirectory(pcsx2-gsrunner)
endif()

#-------------------------------------------------------------------------------
if(NOT IS_SUPPORTED_COMPILER)
	message(WARNING "
*************** UNSUPPORTED CONFIGURATION ***************
You are not compiling PCSX2 with a supported compiler.
It may not even build successfully.
PCSX2 only supports the Clang and MSVC compilers.
No support will be provided, continue at your own risk.
*********************************************************")
endif()

#if(_M_ARM64)
#	message(WARNING "
#*************** UNSUPPORTED CONFIGURATION ***************
#
#Apple Silicon support in PCSX2 is INCOMPLETE. There are
#currently no EE/VU/IOP recompilers, and games will run
#VERY slow. There is no date for completion yet, you
#should set -DCMAKE_OSX_ARCHITECTURES=x86_64 for now,
#unless you want to work on the recompilers.
#
#We also ask that you read https://dont-ship.it/.
#
#*********************************************************")
#endif()
