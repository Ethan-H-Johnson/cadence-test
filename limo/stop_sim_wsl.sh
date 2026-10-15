#!/usr/bin/env bash
# Stop the Gazebo LIMO sim started by run_sim_wsl.sh:
#   wsl -d Ubuntu-20.04-ROS -- bash /mnt/c/.../limo/stop_sim_wsl.sh
pkill -INT -f "bin/roslaunch limo_cadence_bridge" || true
for _ in $(seq 20); do
  pgrep -f "gzserver|bridge_node.py|rosmaster" >/dev/null || { echo stopped; exit 0; }
  sleep 1
done
pkill -9 -f "gzserver|bridge_node.py|rosmaster|bin/roslaunch" || true
echo "stopped (forced)"
