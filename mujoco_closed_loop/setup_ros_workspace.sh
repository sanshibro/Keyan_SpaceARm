#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT/ros_ws/src" "$ROOT/runs"
link_if_missing() {
  local target="$1" path="$2"
  if [[ -e "$path" || -L "$path" ]]; then
    [[ "$(readlink -- "$path")" == "$target" ]] || {
      echo "Existing path differs; left unchanged: $path" >&2
      exit 1
    }
  else
    ln -s -- "$target" "$path"
  fi
}
link_if_missing ../../vendor/FAST_LIO "$ROOT/ros_ws/src/fast_lio"
link_if_missing ../../vendor/livox_ros_driver/livox_ros_driver "$ROOT/ros_ws/src/livox_ros_driver"
link_if_missing /opt/ros/noetic/share/catkin/cmake/toplevel.cmake "$ROOT/ros_ws/src/CMakeLists.txt"
echo 'ROS source links prepared. Build with ROS Noetic: cd mujoco_closed_loop/ros_ws && catkin_make -j1'
