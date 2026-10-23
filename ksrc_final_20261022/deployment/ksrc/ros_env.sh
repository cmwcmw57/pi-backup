#!/usr/bin/env bash
# ================================================================
# ros_env.sh
# 설명: Pi의 기존 ROS 2 Jazzy·ORB 라이브러리를 적용하고 전달받은 프로세스만 실행한다.
# 사용법: bash ros_env.sh python ros_inputs.py --config local/real.json
# ================================================================
set -eo pipefail
KSRC_DOMAIN="${ROS_DOMAIN_ID:-76}"
source /opt/ros/jazzy/setup.bash
source /home/pi/handheld_mapping/ros2_ws/install/orbslam3_ros2/share/orbslam3_ros2/local_setup.bash
export ROS_DOMAIN_ID="$KSRC_DOMAIN" RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH="/home/pi/handheld_mapping/runtime/install/lib:/home/pi/handheld_mapping/runtime/prefix/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export FASTRTPS_DEFAULT_PROFILES_FILE="/home/pi/handheld_mapping/ros2_ws/install/orbslam3_ros2/share/orbslam3_ros2/config/fastdds_stereo_local.xml"
exec "$@"
