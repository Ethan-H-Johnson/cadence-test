#!/usr/bin/env bash
# Fetch AgileX's ROS 2 limo_description (BSD-3-Clause) into ros2/src/agilex, at the commit
# the rover's driver was built from (humble branch, dcc5a86). Meshes stay out of git.
set -euo pipefail

REPO=https://github.com/agilexrobotics/limo_ros2
COMMIT=dcc5a8637f83236ed6778af86ad826e46a20cd92
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/ros2/src/agilex"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

git -C "$WORK" init -q
git -C "$WORK" remote add origin "$REPO"
git -C "$WORK" fetch -q --depth 1 origin "$COMMIT"
git -C "$WORK" checkout -q FETCH_HEAD

rm -rf "$DEST"
mkdir -p "$DEST"
cp -r "$WORK/limo_description" "$DEST/"
echo "$REPO @ $COMMIT" > "$DEST/SOURCE"
echo "fetched into $DEST:"; ls "$DEST"
