set(CMAKE_CXX_STANDARD 11)

# Define the library
add_library(autograd STATIC src/value.cpp src/operators.cpp)

# Specify the include directory for this library's headers
target_include_directories(autograd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
