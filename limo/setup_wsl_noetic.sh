#!/usr/bin/env bash
# Install ROS Noetic + Gazebo 11 and build the LIMO workspace, inside a WSL Ubuntu 20.04.
#
# The WSL store no longer offers Ubuntu 20.04; import Ubuntu's official base image instead
# (see README), run wsl_bootstrap_root.sh as root, then this script as the new user:
#   bash /mnt/c/Users/<you>/.../cadence-test/limo/setup_wsl_noetic.sh
#
# Noetic matches the Nano LIMO's ROS1 Melodic interfaces (same topics, launch files and
# Gazebo plugins). Noetic is end-of-life (May 2025) but its packages remain installable.
# Tested 2026-10-15 in WSL2 on Windows 11 with ubuntu-base 20.04.5.
set -euo pipefail

. /etc/os-release
if [ "$VERSION_CODENAME" != "focal" ]; then
  echo "Expected Ubuntu 20.04 (focal), found $PRETTY_NAME. ROS Noetic needs focal." >&2
  exit 1
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEBIAN_FRONTEND=noninteractive

sudo -E apt-get update
sudo -E apt-get install -y curl git lsb-release software-properties-common
sudo -E add-apt-repository -y universe

# ROS apt source (the ros-apt-source package replaces the expired 2025 signing key setup)
if ! dpkg -s ros-apt-source >/dev/null 2>&1; then
  VER=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
        | grep -F '"tag_name"' | awk -F'"' '{print $4}')
  curl -L -o /tmp/ros-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${VER}/ros-apt-source_${VER}.focal_all.deb"
  sudo dpkg -i /tmp/ros-apt-source.deb
fi
sudo -E apt-get update
# desktop (rviz, rqt) + Gazebo 11 and its ROS plugins; desktop-full adds perception extras unused here
sudo -E apt-get install -y ros-noetic-desktop ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control \
  ros-noetic-ros-control ros-noetic-ros-controllers ros-noetic-joint-state-publisher-gui \
  ros-noetic-rqt-robot-steering ros-noetic-xacro ros-noetic-robot-state-publisher build-essential

[ -d "$HERE/ros/src/agilex/limo_description" ] || bash "$HERE/fetch_limo_sim.sh"

# Build in the Linux filesystem (fast); the packages stay in the repo, linked in.
WS=~/limo_ws
mkdir -p "$WS/src"
for pkg in "$HERE/ros/src/limo_cadence_bridge" "$HERE"/ros/src/agilex/limo_*; do
  ln -sfn "$pkg" "$WS/src/$(basename "$pkg")"
done
set +u  # ROS setup files read unset variables
# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash
set -u
cd "$WS"
catkin_make
grep -q "limo_ws/devel/setup.bash" ~/.bashrc || {
  echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
  echo "source $WS/devel/setup.bash" >> ~/.bashrc
  # mirrored WSL networking hangs on closed localhost ports; see run_sim_wsl.sh
  echo 'export ROS_MASTER_URI="http://$(hostname):11311"' >> ~/.bashrc
}
echo
echo "Done. Open a new WSL shell and run:  roslaunch limo_cadence_bridge sim.launch"
