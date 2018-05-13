project(SequentialSFM)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE "Release") 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
#set(OpenCV_DIR /usr/local/opencv/3.4.0/share/OpenCV)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})



find_package( Ceres REQUIRED )
include_directories( 
	${CERES_INCLUDE_DIRS}
)

add_executable(Reconstruction
	Reconstruction.cpp
)
target_link_libraries(Reconstruction
			${OpenCV_LIBS}
			${CERES_LIBRARIES}
)
