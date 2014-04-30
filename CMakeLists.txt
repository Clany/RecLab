cmake_minimum_required (VERSION 2.8)
project(ReconstructLab)

set(ReconstructLab_VERSION_MAJOR 0)
set(ReconstructLab_VERSION_MINOR 1)

set(MY_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
include_directories(${MY_INCLUDE_DIR})

set(ENABLE_KINECT 0)
if(MSVC AND NOT MSVC_VERSION LESS 1800)  
  option(WITH_KINECT "Include Microsoft Kinect SDK support")
  if(WITH_KINECT)
    include_directories($ENV{KINECTSDK_DIR}/inc)
	link_directories($ENV{KINECTSDK_DIR}/lib/x86)
    find_library(KINECT_LIB Kinect10 $ENV{KINECTSDK_DIR}/lib/x86)
    if(NOT KINECT_LIB)
  	  message(FATAL_ERROR "Kinect SDK not found")
    endif()
	set(ENABLE_KINECT 1)
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
    message(FATAL_ERROR "GCC version need to be at least 4.9")
  endif()
  set(CMAKE_CXX_FLAGS "-std=c++1y -fopenmp -O3")
else()
  message(FATAL_ERROR "No supported compiler")
endif()

configure_file (
  "${MY_INCLUDE_DIR}/reclab_config.h.in"
  "${MY_INCLUDE_DIR}/reclab_config.h"
)
  
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

add_subdirectory(src)