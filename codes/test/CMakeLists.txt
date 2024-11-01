# Set the minimum required version and project name
cmake_minimum_required(VERSION 3.0)
project(test)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)

# Add the library directory as a dependency
add_subdirectory(../lib ${CMAKE_BINARY_DIR}/lib)

# Include the header files directory, specifying the subdirectory
include_directories(${CMAKE_SOURCE_DIR}/../lib/include/autograd)

# Define the main executable
add_executable(test src/main.cpp)

# Link with the libraries
target_link_libraries(test PRIVATE autograd)
