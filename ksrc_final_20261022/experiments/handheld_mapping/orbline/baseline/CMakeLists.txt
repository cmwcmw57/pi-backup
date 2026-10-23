# ================================================================
# CMakeLists.txt
# 설명: 기존 native ORB-SLAM3로 같은 offline stereo 입력을 비교하는 실행기를 빌드한다.
# 사용법: cmake -S experiments/handheld_mapping/orbline/baseline -B outputs/handheld_mapping/orb3_comparison_build -DORB3_ROOT=...
# ================================================================
cmake_minimum_required(VERSION 3.16)
project(rover_orb3_comparison LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
option(ROVER_BUILD_IMU_COMPARE "Build the separate offline IMU_STEREO comparison runner" OFF)
add_executable(rover_orb3_compare ../runner.cpp)
target_compile_definitions(rover_orb3_compare PRIVATE ROVER_POINT_BASELINE EIGEN_HAS_CXX17_OVERALIGN=0)
target_compile_options(rover_orb3_compare PRIVATE -O1 -g0 -march=native)
target_include_directories(rover_orb3_compare PRIVATE ${ORB3_ROOT}/include ${ORB3_ROOT}/include/CameraModels
  ${ORB3_ROOT} ${ORB3_ROOT}/Thirdparty/Sophus ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${Pangolin_INCLUDE_DIRS})
target_link_libraries(rover_orb3_compare ${ORB3_ROOT}/lib/libORB_SLAM3.so ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
if(ROVER_BUILD_IMU_COMPARE)
  enable_testing()
  add_executable(rover_imu_input_probe ../../../../tests/handheld_mapping/native/inertial_runner_probe.cpp)
  target_include_directories(rover_imu_input_probe PRIVATE ..)
  add_test(NAME inertial_input COMMAND rover_imu_input_probe)
  add_executable(rover_orb3_imu_compare ../runner.cpp)
  target_compile_definitions(rover_orb3_imu_compare PRIVATE ROVER_POINT_BASELINE ROVER_IMU_ONLY EIGEN_HAS_CXX17_OVERALIGN=0)
  target_compile_options(rover_orb3_imu_compare PRIVATE -O1 -g0 -march=native)
  target_include_directories(rover_orb3_imu_compare PRIVATE ${ORB3_ROOT}/include ${ORB3_ROOT}/include/CameraModels
    ${ORB3_ROOT} ${ORB3_ROOT}/Thirdparty/Sophus ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${Pangolin_INCLUDE_DIRS})
  target_link_libraries(rover_orb3_imu_compare ${ORB3_ROOT}/lib/libORB_SLAM3.so ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
endif()
