cmake_minimum_required(VERSION 3.2)

set(CMAKE_CXX_STANDARD 11)
project(SDK_DEMO)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories( ${CMAKE_CURRENT_SOURCE_DIR}/lib)

set (opencv_libs opencv_highgui.so opencv_video.so opencv_imgproc.so opencv_core.so opencv_videoio.so opencv_imgcodecs.so)
add_executable(test_trt test.cpp)
target_link_libraries(test_trt TRTEngine ${opencv_libs})
