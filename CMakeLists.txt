cmake_minimum_required(VERSION 3.20)

project(opencv_packt_book LANGUAGES C CXX)

set(CMAKE_CXX_VERSION 17)

find_package(OpenCV 4.0.0 REQUIRED)

add_subdirectory(chapter2)
add_subdirectory(chapter3)
add_subdirectory(chapter4)
add_subdirectory(chapter5)
