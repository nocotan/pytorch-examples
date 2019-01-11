cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dataset)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++17 -I/usr/local/Cellar/opencv/3.4.3/include/")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED )


add_executable(dataset dataset.cc)
target_link_libraries(dataset "${TORCH_LIBRARIES}")
target_link_libraries(dataset "${OpenCV_LIBS}")
set_property(TARGET dataset PROPERTY CXX_STANDARD 11)
