"""The Cadence bridge on the rover. Start the chassis driver (with motion enabled) and the lidar
driver first. Lift the wheels off the ground for the first test.

    ros2 launch limo_cadence_bridge robot.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package="limo_cadence_bridge", executable="bridge", output="screen",
             parameters=[{"max_linear": 0.3, "max_angular": 1.2, "reset_world": ""}]),
    ])
