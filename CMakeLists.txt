cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Project name
project(orb)

# Set version
set(ORB_MAJOR_VERSION 1)
set(ORB_MINOR_VERSION 0)
set(ORB_PATCH_VERSION 0)
set(ORB_VERSION ${ORB_MAJOR_VERSION}.${ORB_MINOR_VERSION}.${ORB_PATCH_VERSION})

# Set C++ standard
# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -std=c++14 -fPIC")

# Offer the user the choice of overriding the installation directories
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH
  "Installation directory for header files")
if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/CMake/orb)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
  "Installation directory for CMake files")

# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

# make sure we use Release and warn otherwise
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()
if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  message(WARNING "CMAKE_BUILD_TYPE not set to 'Release'. Performance may be terrible.")
else()
  message(STATUS "Building with build type '${CMAKE_BUILD_TYPE}'")
endif()

# require OpenCV
find_package( OpenCV 3.4.0 REQUIRED )
include_directories(BEFORE ${OpenCV_INCLUDE_DIRS} )

# Arm / X86 optimize flag
if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "arm*")
  message(STATUS "ARM processor detected, will attempt to use NEON.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
else()
  message(STATUS "Assuming SSE instructions available.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mssse3")
endif()

if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "arm*")
  message(STATUS "ARM processor detected, will attempt to use NEON.")
  add_definitions(-mfpu=neon -DHAVE_OPENCV -Wall -pedantic -std=c++0x -fPIC)
else()
  message(STATUS "Assuming SSE instructions available.")
  add_definitions(-mssse3 -DHAVE_OPENCV -Wall -pedantic -std=c++0x -fPIC)
endif()

# set up the output tree
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT APPLE)
  # The clang compiler (on osx) is somehow much more strict
  # than the compilers on ubuntu and so this does not seem
  # possible on OSX just yet.
  add_definitions( -Werror )
endif()


# build ORB
include_directories(include)
add_library(${PROJECT_NAME} STATIC src/orb.cc)


# and link it
target_link_libraries(${PROJECT_NAME} PUBLIC
  ${OpenCV_LIBS})


# installation if required
install(TARGETS ${PROJECT_NAME} 
  EXPORT orbTargets
  ARCHIVE DESTINATION ${INSTALL_LIB_DIR})

install(DIRECTORY include/ DESTINATION 
  ${INSTALL_INCLUDE_DIR} COMPONENT 
  dev FILES_MATCHING PATTERN "*.h")

# installation is invoked in the individual modules...
export (TARGETS 
    orb
    FILE "${PROJECT_BINARY_DIR}/orbTargets.cmake")
export ( PACKAGE orb )

# Create the orbConfig.cmake and orbConfigVersion files
file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
   "${INSTALL_INCLUDE_DIR}")

# ... for the build tree
set(CONF_INCLUDE_DIRS 
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_BINARY_DIR}"
)

configure_file(cmake/orbConfig.cmake.in
  "${PROJECT_BINARY_DIR}/orbConfig.cmake" @ONLY)

# ... for the install tree
set(CONF_INCLUDE_DIRS "\${ORB_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(cmake/orbConfig.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/orbConfig.cmake" @ONLY)

# ... for both
configure_file(cmake/orbConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}/orbConfigVersion.cmake" @ONLY)
 
# Install the orbConfig.cmake and orbConfigVersion.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/orbConfig.cmake"
  "${PROJECT_BINARY_DIR}/orbConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}")

# Install the export set for use with the install-tree
install(EXPORT orbTargets DESTINATION
  "${INSTALL_CMAKE_DIR}")