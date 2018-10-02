cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example_app)

find_package(Torch REQUIRED)

add_executable(example_app example_app.cc)
target_link_libraries(example_app "${TORCH_LIBRARIES}")
set_property(TARGET example_app PROPERTY CXX_STANDARD 11)
