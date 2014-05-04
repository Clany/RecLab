set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

set(SRC_LIST reclab_app.cpp main.cpp)

set(CORE_SRC_LIST
	core/reclab.cpp	
	core/init_matcher.cpp
	core/optical_flow_matcher.cpp
	core/rich_feature_matcher.cpp
	core/bundle_adjust.cpp
)

set(UTIL_SRC_LIST
	utilities/init_grabber.cpp
)

if(MSVC)
  file(GLOB HEADER_LIST ${MY_INCLUDE_DIR}/*.h)
  file(GLOB HEADER_LIST ${HEADER_LIST} ${MY_INCLUDE_DIR}/*.hpp)
  file(GLOB HEADER_LIST ${HEADER_LIST} ${MY_INCLUDE_DIR}/*/*.h)
  file(GLOB HEADER_LIST ${HEADER_LIST} ${MY_INCLUDE_DIR}/*/*.hpp)
  add_executable(RecLab ${SRC_LIST} ${CORE_SRC_LIST} ${UTIL_SRC_LIST} ${HEADER_LIST})
  target_link_libraries(RecLab ${OpenCV_LIBS} ${SSBA_LIBS})
else()
  add_executable(RecLab ${SRC_LIST} ${CORE_SRC_LIST} ${UTIL_SRC_LIST})
  target_link_libraries(RecLab SSBA_V3D SSBA_COLAMD ${OpenCV_LIBS})
endif()