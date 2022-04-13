cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Set project name
project( orb_example )

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

#####
# Change the default build type from Debug to Release, while still
# supporting overriding the build type.
#
# The CACHE STRING logic here and elsewhere is needed to force CMake
# to pay attention to the value of these variables.
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type specified; defaulting to CMAKE_BUILD_TYPE=Release.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
else(NOT CMAKE_BUILD_TYPE)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("==========================================================================================")
        message(STATUS "Build type: Debug. Performance will be terrible!")
        message(STATUS "Add -DCMAKE_BUILD_TYPE=Release to the CMake command line to get an optimized build.")
        message("==========================================================================================")
    endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
endif(NOT CMAKE_BUILD_TYPE)

find_package( OpenCV 3.4.0 REQUIRED )
include_directories( BEFORE ${OpenCV_INCLUDE_DIRS})
message( STATUS "Found OpenCV: ${OpenCV_INCLUDE_DIRS} ${OpenCV_LIBS}")

find_package( orb REQUIRED )
include_directories( BEFORE ${ORB_INCLUDE_DIRS} )
message( STATUS "Found ORB: ${ORB_INCLUDE_DIRS} ${ORB_LIBRARIES}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR} )

add_executable( demo demo.cc  )
target_link_libraries( demo PRIVATE ${OpenCV_LIBS} ${ORB_LIBRARIES})