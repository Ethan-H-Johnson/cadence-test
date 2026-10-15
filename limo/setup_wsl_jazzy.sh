#!/usr/bin/env bash
# Install ROS 2 Jazzy + Gazebo Harmonic and build the ROS 2 LIMO workspace in a WSL Ubuntu
# 24.04 distro, matching the rover (Jetson Orin Nano Super, Ubuntu 24.04, ROS 2 Jazzy).
#
# The distro is imported from Ubuntu's official base image and bootstrapped with
# wsl_bootstrap_root.sh (see README); then, as the new user:
#   bash /mnt/c/Users/<you>/.../cadence-test/limo/setup_wsl_jazzy.sh
set -euo pipefail

. /etc/os-release
if [ "$VERSION_CODENAME" != "noble" ]; then
  echo "Expected Ubuntu 24.04 (noble), found $PRETTY_NAME. ROS 2 Jazzy needs noble." >&2
  exit 1
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEBIAN_FRONTEND=noninteractive

sudo -E apt-get update
sudo -E apt-get install -y curl git software-properties-common
sudo -E add-apt-repository -y universe
if ! dpkg -s ros2-apt-source >/dev/null 2>&1; then
  VER=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
        | grep -F '"tag_name"' | awk -F'"' '{print $4}')
  curl -L -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${VER}/ros2-apt-source_${VER}.noble_all.deb"
  sudo dpkg -i /tmp/ros2-apt-source.deb
fi
sudo -E apt-get update
# ros-gz brings Gazebo Harmonic, the simulator paired with Jazzy
sudo -E apt-get install -y ros-jazzy-desktop ros-jazzy-ros-gz ros-jazzy-xacro \
  ros-jazzy-robot-state-publisher python3-colcon-common-extensions build-essential

[ -d "$HERE/ros2/src/agilex/limo_description" ] || bash "$HERE/fetch_limo_ros2.sh"

# Build in the Linux filesystem; the packages stay in the repo, linked in.
WS=~/limo_ros2_ws
mkdir -p "$WS/src"
for pkg in "$HERE"/ros2/src/limo_* "$HERE"/ros2/src/agilex/limo_*; do
  [ -d "$pkg" ] && ln -sfn "$pkg" "$WS/src/$(basename "$pkg")"
done
set +u  # ROS setup files read unset variables
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
set -u
cd "$WS"
colcon build --symlink-install
grep -q "limo_ros2_ws/install/setup.bash" ~/.bashrc || {
  echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
  echo "source $WS/install/setup.bash" >> ~/.bashrc
  echo "export ROS_DOMAIN_ID=26  # the rover's domain" >> ~/.bashrc
  echo "export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST" >> ~/.bashrc
}
echo
echo "Done. Start the sim with run_sim_jazzy.sh (see README)."
