#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#Tiny Differentiable Simulator or TDS

cmake_minimum_required(VERSION 2.4.3)
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # For LLVM tooling
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)

cmake_policy(SET CMP0042 NEW)

cmake_policy(SET CMP0017 NEW)
 if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)
    endif(COMMAND cmake_policy)
    
#this line has to appear before 'PROJECT' in order to be able to disable incremental linking
SET(MSVC_INCREMENTAL_DEFAULT ON)

PROJECT(DIFF_PHYSICS)

#The core library consists of those headers
FILE(GLOB TDS_HDRS "*.h" )

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3)
if (Eigen3_FOUND)
    message("FOUND EIGEN")
    add_definitions(-DUSE_EIGEN)
    include_directories(${EIGEN3_INCLUDE_DIRS})
endif (Eigen3_FOUND)


include_directories(src)


IF(WIN32)
	SET(BUILD_SHARED_LIBS OFF CACHE BOOL "Shared Libs" FORCE)
ELSE(WIN32)
	SET(BUILD_SHARED_LIBS ON CACHE BOOL "Shared Libs" FORCE)
ENDIF(WIN32)




ADD_DEFINITIONS(-DSTATIC_LINK_SPD_PLUGIN)



set(ENOKI_AUTODIFF ON CACHE BOOL "Use enoki with autodiff" FORCE)
set(ENOKI_CUDA ON CACHE BOOL "Use enoki with CUDA" FORCE)
add_subdirectory(third_party/enoki)
enoki_set_compile_flags()
enoki_set_native_flags()
get_directory_property(ENOKI_COMPILE_OPTIONS     COMPILE_OPTIONS)
get_directory_property(ENOKI_COMPILE_DEFINITIONS COMPILE_DEFINITIONS)
set_property(DIRECTORY . PROPERTY COMPILE_OPTIONS     ${ENOKI_COMPILE_OPTIONS})
set_property(DIRECTORY . PROPERTY COMPILE_DEFINITIONS ${ENOKI_COMPILE_DEFINITIONS})
set(ENOKI_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/enoki/include)
include_directories(${ENOKI_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD_LIBRARIES ${CMAKE_CXX_STANDARD_LIBRARIES})
# set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS})
# set(CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS})





SUBDIRS(third_party/cpp_base64)

SUBDIRS(examples)

OPTION(BUILD_PYTHON_PLUGIN "Set when you want to build pydiffphys (Python bindings for tiny-differentiable-simulator)" OFF)
IF(BUILD_PYTHON_PLUGIN)
  SUBDIRS(python)
ENDIF()
