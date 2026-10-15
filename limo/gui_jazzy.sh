#!/usr/bin/env bash
# Open a Gazebo window on the running sim (a separate process: closing or crashing it leaves
# the simulation running). Started by run_sim_jazzy.sh gui:=true, or on its own:
#   wsl -d Ubuntu-24.04-ROS2 -- bash /mnt/c/.../limo/gui_jazzy.sh
# Renders on the GPU through WSL's D3D12 driver; GUI_GPU=0 uses software rendering, which
# drew at ~1.8 CPU cores and felt frozen.
set +u
source /opt/ros/jazzy/setup.bash
export QT_QPA_PLATFORM=xcb   # WSLg's Wayland backend left the window blank
if [ "${GUI_GPU:-1}" = 1 ] && [ -e /dev/dxg ]; then
  export GALLIUM_DRIVER=d3d12
  export MESA_D3D12_DEFAULT_ADAPTER_NAME="${SIM_GPU_NAME:-NVIDIA}"
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"  # WSL GPU driver first
fi
exec gz sim -g
