#!/usr/bin/env bash
# Stop the sim started by run_sim_jazzy.sh.
pkill -INT -f "ros2 launch limo_sim" || true
pkill -INT -f "gz sim -g" || true
for _ in $(seq 20); do
  pgrep -f "gz sim|parameter_bridge|limo_cadence_bridge/bridge|robot_state_publisher" >/dev/null || { echo stopped; exit 0; }
  sleep 1
done
pkill -9 -f "gz sim|parameter_bridge|limo_cadence_bridge/bridge|robot_state_publisher|ros2 launch limo_sim" || true
echo "stopped (forced)"
