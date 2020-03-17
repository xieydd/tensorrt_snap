cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

set(OpenCV_DIR ${CMNAKE_CURRENT_SOURCE_DIR}/../opencv-3.4.2_install/share/OpenCV)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
cuda_add_executable(test_trt test.cpp)
target_link_libraries(test_trt TRTEngine ${OpenCV_LIBS})
#install(TARGETS test_trt DESTINATION bin)
