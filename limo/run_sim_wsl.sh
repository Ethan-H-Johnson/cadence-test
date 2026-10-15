#!/usr/bin/env bash
echo "run_sim_wsl.sh starting $(date +%T)"
# Start the Gazebo LIMO + Cadence bridge inside the WSL distro, non-interactively.
#   wsl -d Ubuntu-20.04-ROS -- bash /mnt/c/.../limo/run_sim_wsl.sh [roslaunch args, e.g. gui:=false]
# Log: ~/sim.log. Stop with: wsl -d Ubuntu-20.04-ROS -- pkill -f roslaunch
set -e
set +u
source /opt/ros/noetic/setup.bash
source ~/limo_ws/devel/setup.bash
# With WSL's mirrored networking a closed port on 127.0.0.1 drops connections instead of
# refusing them, so roslaunch's "is a master running?" check on localhost:11311 hangs ~2 min.
# The hostname resolves to 127.0.1.1, which refuses at once.
export ROS_MASTER_URI="http://$(hostname):11311"
# Gazebo 11 tries the retired online model database for unknown models; never wait on it
export GAZEBO_MODEL_DATABASE_URI=""
exec roslaunch limo_cadence_bridge sim.launch "$@"
