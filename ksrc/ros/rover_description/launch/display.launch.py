from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    root=Path(get_package_share_directory('rover_description'))
    description=(root/'urdf/rover.urdf').read_text()
    return LaunchDescription([
        Node(package='robot_state_publisher',executable='robot_state_publisher',parameters=[{'robot_description':description}]),
        Node(package='joint_state_publisher_gui',executable='joint_state_publisher_gui',parameters=[{'robot_description':description}]),
        Node(package='rviz2',executable='rviz2',arguments=['-d',str(root/'rviz/rover.rviz')]),
    ])
