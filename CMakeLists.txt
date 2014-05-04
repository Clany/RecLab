cmake_minimum_required (VERSION 2.8)
project(RecLab)

set(RecLab_VERSION_MAJOR 0)
set(RecLab_VERSION_MINOR 1)

set(MY_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
include_directories(${MY_INCLUDE_DIR})

set(ENABLE_KINECT 0)
if(MSVC)
  if(MSVC_VERSION LESS 1800)
    message(FATAL_ERROR "MSVC version need to be at least 2013")
  endif()
  set(SSBA_LIBS colamd.lib V3D.lib)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8)
    message(FATAL_ERROR "GCC version need to be at least 4.8")
  endif()
  set(CMAKE_CXX_FLAGS "-std=c++11 -fopenmp -O3")
  add_library(SSBA_V3D STATIC IMPORTED)
  set_property(TARGET SSBA_V3D PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/3rdparty/lib/libV3D.a)
  add_library(SSBA_COLAMD STATIC IMPORTED)
  set_property(TARGET SSBA_COLAMD PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/3rdparty/lib/libcolamd.a)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "CLANG")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.2)
    message(FATAL_ERROR "Clang version need to be at least 3.2")
  endif()
  set(CMAKE_CXX_FLAGS "-std=c++11 -fopenmp -O3")
  set(SSBA_LIBS -lcolamd -lV3D)
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

include_directories(${PROJECT_SOURCE_DIR}/3rdparty/ssba)
link_directories(${PROJECT_SOURCE_DIR}/3rdparty/lib)

add_subdirectory(src)