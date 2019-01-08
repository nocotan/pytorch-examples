cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(simple_mnist)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++17")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(Torch REQUIRED)

add_executable(simple_mnist simple_mnist.cc)
target_link_libraries(simple_mnist "${TORCH_LIBRARIES}")
set_property(TARGET simple_mnist PROPERTY CXX_STANDARD 11)
