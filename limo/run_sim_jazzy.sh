#!/usr/bin/env bash
# Start the Gazebo Harmonic LIMO Pro + Cadence bridge in the Ubuntu-24.04-ROS2 distro.
#   wsl -d Ubuntu-24.04-ROS2 -- bash /mnt/c/.../limo/run_sim_jazzy.sh [gui:=true] [layout:=open|boxes]
# Log: ~/sim.log when started as in the README. Stop with stop_sim_jazzy.sh.
echo "run_sim_jazzy.sh starting $(date +%T)"
set +u  # ROS setup files read unset variables
source /opt/ros/jazzy/setup.bash
source ~/limo_ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=26                          # the rover's domain
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST   # keep sim traffic on this machine
# The lidar is rendered in software (Mesa llvmpipe) by default: stable, at the cost of running
# slower than real time; the controller paces itself by sim time, so its timing stays exact.
# SIM_GPU=1 renders on the GPU through WSL's D3D12 driver (faster, but on 2026-10-15 Gazebo
# segfaulted inside libd3d12core.so after ~5 minutes).
if [ "${SIM_GPU:-0}" = 1 ] && [ -e /dev/dxg ]; then
  export GALLIUM_DRIVER=d3d12
  export MESA_D3D12_DEFAULT_ADAPTER_NAME="${SIM_GPU_NAME:-NVIDIA}"
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"  # WSL GPU driver first
fi
# gui:=true opens the Gazebo window as a separate process (gui_jazzy.sh): the simulation
# keeps running if the window freezes, crashes or is closed.
ARGS=()
GUI=0
for a in "$@"; do
  if [ "$a" = "gui:=true" ]; then GUI=1; else ARGS+=("$a"); fi
done
if [ "$GUI" = 1 ]; then
  ( sleep 8; bash "$(dirname "${BASH_SOURCE[0]}")/gui_jazzy.sh" > ~/gui.log 2>&1 ) &
fi
exec ros2 launch limo_sim sim.launch.py "${ARGS[@]}"
