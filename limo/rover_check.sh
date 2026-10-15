#!/usr/bin/env bash
# Read-only check of the rover, to make the sim match it. Sends no commands and writes
# nothing except ~/cadence_rover/rover.json (the bench result) when Cadence has been copied.
#
# On the rover, with the telemetry driver running in another SSH session (~/limo-telemetry):
#   bash ~/cadence_rover/limo/rover_check.sh
set -u
set +u; source /opt/ros/jazzy/setup.bash 2>/dev/null; set -u

echo "== machine"
tr -d '\0' < /proc/device-tree/model 2>/dev/null; echo
nproc; free -m | awk '/Mem:/ {print "memory MB:", $2}'
python3 --version; python3 -c "import numpy; print('numpy', numpy.__version__)" 2>&1
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-unset} ROS_AUTOMATIC_DISCOVERY_RANGE=${ROS_AUTOMATIC_DISCOVERY_RANGE:-unset}"

echo "== chassis (needs ~/limo-telemetry running)"
timeout 5 ros2 topic echo --once /limo_status 2>&1 | grep -E "motion_mode|control_mode|vehicle_state|battery|error" \
  || echo "no /limo_status: start ~/limo-telemetry first"
echo "motion_mode: 0 = four-wheel differential, 1 = Ackermann, 2 = mecanum"
for t in /odom /imu; do
  echo "-- rate $t"; timeout 6 ros2 topic hz "$t" 2>&1 | grep -m1 "average rate" || echo "no messages"
done
timeout 5 ros2 topic echo --once /imu 2>&1 | grep -A4 "orientation:" | head -5

echo "== lidar adapter"
ls -l /dev/ttyUSB* 2>/dev/null || echo "no /dev/ttyUSB*"
for d in /dev/ttyUSB*; do
  [ -e "$d" ] && udevadm info -q property -n "$d" | grep -E "ID_MODEL=|ID_VENDOR=|ID_SERIAL="
done
ls /dev/ | grep -i -E "ydlidar|limo|lidar" || true

echo "== Cadence bench (this computer's time per brain step)"
if [ -d ~/cadence_rover/limo/limo_brain ]; then
  cd ~/cadence_rover/limo && python3 -m limo_brain.bench --save ~/cadence_rover/rover.json | grep -E "median|p95|max"
  echo "saved ~/cadence_rover/rover.json; copy it to the PC for --calibration"
else
  echo "skipped: copy Cadence first (see limo/README.md, 'Calibrate on the rover')"
fi
