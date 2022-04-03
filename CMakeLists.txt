cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(orb)


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


# require OpenCV
find_package( OpenCV REQUIRED )
include_directories(BEFORE ${OpenCV_INCLUDE_DIRS} )


# set up the output tree
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# build ORB
include_directories(include)
add_library(${PROJECT_NAME} STATIC src/orb.cc)


# and link it                   
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})


# installation if required
install(TARGETS ${PROJECT_NAME} 
	EXPORT orbTargets 
	ARCHIVE DESTINATION ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} # ${INSTALL_LIB_DIR} user/local no permission
	)


# installation is invoked in the individual modules...
export (TARGETS 
    orb
    FILE "${PROJECT_BINARY_DIR}/orbTargets.cmake")
export ( PACKAGE orb )

set(CONF_INCLUDE_DIRS 
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_BINARY_DIR}"
)
configure_file(cmake/orbConfig.cmake.in
  "${PROJECT_BINARY_DIR}/orbConfig.cmake" @ONLY)
#set(CONF_INCLUDE_DIRS "\${BRISK_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(cmake/orbConfig.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/orbConfig.cmake" @ONLY)

install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/orbConfig.cmake"
  DESTINATION "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}") # ${INSTALL_CMAKE_DIR} user/local no permission

