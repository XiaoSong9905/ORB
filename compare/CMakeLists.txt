cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Set project name
project( orb_compare )

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# options
option (USE_SYSTEM_BRISK
        "Use brisk via find_package rather than downloading it as part of okvis" OFF) 

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

find_package( OpenCV REQUIRED )
include_directories( BEFORE ${OpenCV_INCLUDE_DIRS})

find_package( orb REQUIRED )
include_directories( ${ORB_INCLUDE_DIRS} )
message( STATUS "Found ORB: ${ORB_INCLUDE_DIRS}")

# brisk
include(ExternalProject)
if(${USE_SYSTEM_BRISK})
  find_package( brisk 2 REQUIRED )
  include_directories(BEFORE ${BRISK_INCLUDE_DIRS})
  message(STATUS "Using system brisk. Found at ${BRISK_INCLUDE_DIRS}.")
else()
  ExternalProject_Add(brisk_external
    URL "https://www.doc.ic.ac.uk/~sleutene/software/brisk-2.0.3.zip"    
    INSTALL_DIR ${CMAKE_BINARY_DIR}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/brisk
    BUILD_IN_SOURCE 0
    BUILD_COMMAND make -j${N_CORES}
    INSTALL_COMMAND make install
  )
  add_library(brisk STATIC IMPORTED)
  add_library(agast STATIC IMPORTED)
  set(BRISK_LIBRARIES brisk agast)
  set_target_properties(brisk PROPERTIES IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/lib/libbrisk.a)
  set_target_properties(agast PROPERTIES IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/lib/libagast.a)
  add_dependencies(brisk brisk_external)
  add_dependencies(agast brisk_external)

  # install it once built
  install(FILES ${CMAKE_BINARY_DIR}/lib/libbrisk.a ${CMAKE_BINARY_DIR}/lib/libagast.a 
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
  install(DIRECTORY ${CMAKE_BINARY_DIR}/include/brisk/ 
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include/brisk/)
  install(DIRECTORY ${CMAKE_BINARY_DIR}/lib/CMake/brisk/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/CMake/brisk/)
endif()

add_executable( compare compare.cc )
target_link_libraries( compare ${OpenCV_LIBS} ${BRISK_LIBRARIES} orb )