# ================================================================
# CMakeLists.txt
# 설명: 기존 ORB 설치와 분리해 ORB-LINE과 offline runner를 빌드한다.
# 사용법: bash tools/build_orbline.sh
# ================================================================
cmake_minimum_required(VERSION 3.16)
project(rover_orbline LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc features2d calib3d highgui imgcodecs)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(OpenSSL REQUIRED)
get_filename_component(ROVER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(LINE_ROOT "${ROVER_ROOT}/external/ORB-LINE-SLAM")
add_compile_options(-O1 -g0 -march=native -include "${CMAKE_CURRENT_SOURCE_DIR}/compat.h")
add_definitions(-DCOMPILEDWITHC11 -DEIGEN_HAS_CXX17_OVERALIGN=0)
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${Pangolin_INCLUDE_DIRS}
  "${LINE_ROOT}" "${LINE_ROOT}/include" "${LINE_ROOT}/include/CameraModels"
  "${LINE_ROOT}/Thirdparty/line_descriptor/include")
file(GLOB DBOW_SOURCE "${LINE_ROOT}/Thirdparty/DBoW2/DBoW2/*.cpp" "${LINE_ROOT}/Thirdparty/DBoW2/DUtils/*.cpp")
add_library(rover_line_dbow SHARED ${DBOW_SOURCE})
target_link_libraries(rover_line_dbow ${OpenCV_LIBS})
file(GLOB_RECURSE LINE_DESCRIPTOR_SOURCE "${LINE_ROOT}/Thirdparty/line_descriptor/src/*.cpp")
add_library(rover_line_descriptor SHARED ${LINE_DESCRIPTOR_SOURCE})
target_link_libraries(rover_line_descriptor ${OpenCV_LIBS})
add_subdirectory("${LINE_ROOT}/Thirdparty/g2o" g2o-build)
target_compile_options(g2o PRIVATE -O1 -g0 -std=c++14)
file(GLOB CORE_SOURCE "${LINE_ROOT}/src/*.cc" "${LINE_ROOT}/src/*.cpp" "${LINE_ROOT}/src/CameraModels/*.cpp")
add_library(rover_orbline_core SHARED ${CORE_SOURCE})
target_link_libraries(rover_orbline_core ${OpenCV_LIBS} ${Pangolin_LIBRARIES} rover_line_dbow rover_line_descriptor g2o Boost::serialization OpenSSL::Crypto)
add_executable(rover_orbline runner.cpp)
target_link_libraries(rover_orbline rover_orbline_core)
