"""Gazebo Harmonic LIMO Pro in the arena, with the rover's topic names, plus the Cadence bridge.

    ros2 launch limo_sim sim.launch.py              # headless
    ros2 launch limo_sim sim.launch.py gui:=true    # with the Gazebo window (WSLg)
"""

import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    sim = get_package_share_directory("limo_sim")
    description = get_package_share_directory("limo_description")
    gui = LaunchConfiguration("gui")
    world = LaunchConfiguration("world")
    sim_time = {"use_sim_time": True}

    def robot_nodes(context: LaunchContext):
        # The URDF goes to the spawner as a string: waiting for it on /robot_description
        # stalled for minutes under WSL's mirrored networking.
        urdf = xacro.process_file(
            os.path.join(sim, "urdf", "limo_pro.urdf.xacro"),
            mappings={"scan_hz": LaunchConfiguration("scan_hz").perform(context),
                      "body_mesh": LaunchConfiguration("body_mesh").perform(context)},
        ).toxml()
        return [
            Node(package="robot_state_publisher", executable="robot_state_publisher",
                 parameters=[{"robot_description": urdf}, sim_time]),
            Node(package="ros_gz_sim", executable="create", output="screen",
                 arguments=["-string", urdf, "-name", "limo", "-z", "0.01"]),
        ]

    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="false"),
        DeclareLaunchArgument("layout", default_value="open",
                              description="open (walls only) or boxes; match the controller's --layout"),
        DeclareLaunchArgument("world", default_value=[sim, "/worlds/arena_",
                                                      LaunchConfiguration("layout"), ".sdf"]),
        DeclareLaunchArgument("scan_hz", default_value="10"),
        DeclareLaunchArgument("body_mesh", default_value="false",
                              description="draw AgileX's body shell (the lidar then sees it)"),
        DeclareLaunchArgument("bridge", default_value="true", description="start the Cadence TCP bridge"),
        # meshes are referenced by file:// paths; this lets Gazebo also find package:// ones
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", os.path.dirname(description)),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")),
            launch_arguments={
                # -r: run at once; -s: server only unless gui:=true
                "gz_args": PythonExpression(["'-r ' + ('' if '", gui, "' == 'true' else '-s --headless-rendering ') + '", world, "'"]),
                "on_exit_shutdown": "true",
            }.items(),
        ),
        OpaqueFunction(function=robot_nodes),
        Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
             parameters=[{"config_file": os.path.join(sim, "config", "bridge.yaml")}, sim_time]),
        Node(package="limo_cadence_bridge", executable="bridge", output="screen",
             condition=IfCondition(LaunchConfiguration("bridge")),
             parameters=[{"reset_world": "cadence_arena"}, sim_time]),
    ])
