cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Set project name
project( orb_example )

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# set debug
set(CMAKE_BUILD_TYPE Rebug)

# opencv package
find_package( OpenCV 3.4.0 REQUIRED )
include_directories( BEFORE ${OpenCV_INCLUDE_DIRS})
message( STATUS "Found OpenCV: ${OpenCV_INCLUDE_DIRS} ${OpenCV_LIBS}")

# build everything
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR} )

include_directories( ../include )
add_executable( demo demo.cc ../src/orb.cc )
target_link_libraries( demo PRIVATE ${OpenCV_LIBS} )