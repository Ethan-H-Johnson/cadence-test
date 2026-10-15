#!/usr/bin/env bash
# Fetch AgileX's LIMO Gazebo packages (BSD-3-Clause) into ros/src/agilex at a pinned commit.
# The meshes are ~100 MB, so they stay out of git; rerun this script to restore them.
set -euo pipefail

REPO=https://github.com/agilexrobotics/ugv_gazebo_sim
COMMIT=27633a956c845903ee630538afeb17fe70afdd84   # master, 2026-06-03
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/ros/src/agilex"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

git -C "$WORK" init -q
git -C "$WORK" remote add origin "$REPO"
# Sparse (cone) checkout of the limo folder only: the repository also holds files whose names
# differ only by case (scout meshes), which break checkouts on Windows.
git -C "$WORK" sparse-checkout set limo  # no leading slash: Git Bash would rewrite it
git -C "$WORK" fetch -q --depth 1 origin "$COMMIT"
git -C "$WORK" checkout -q FETCH_HEAD

rm -rf "$DEST"
mkdir -p "$DEST"
cp -r "$WORK/limo/limo_description" "$WORK/limo/limo_gazebo_sim" "$DEST/"
echo "$REPO @ $COMMIT" > "$DEST/SOURCE"
echo "LIMO sim packages in $DEST:"
ls "$DEST"
