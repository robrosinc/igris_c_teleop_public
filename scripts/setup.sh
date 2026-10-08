#!/usr/bin/env bash
# Fresh-clone setup for igris_c_gmr, end to end:
#   1. submodules: gmr, vr_link (vr_link's own nested igris_c_sdk is left out: this repo
#      builds the SDK from its top-level thirdparty/igris_c_sdk pin, see install_sdk_wheel.sh),
#      igris_c_sdk with its nested cyclonedds / cyclonedds-cxx / pybind11
#   2. pixi install                           (RoboStack ROS 2 + GMR deps + editable vr_link)
#   3. task wheel in thirdparty/igris_c_sdk   (CMake build + wheel under the SDK's own pixi env)
#   4. pixi run install-sdk                   (pip install the wheel into .pixi/envs/default)
#   5. pixi run check-env                     (SDK import path under .pixi/envs, not shadowed)
#
# Run it from a plain shell, NOT under `pixi run`/`pixi shell` of this repo: step 3 needs the
# host ROS 2 Jazzy (ament_cmake, rosidl_default_generators, std_msgs, sensor_msgs) sourced,
# and an activated gmr env would put RoboStack's ROS and this env's Python in front of it
# during the CMake configure. Only steps 2, 4 and 5 enter the gmr pixi env.
#
# Host prerequisites (not installed here): ROS 2 Jazzy sourced, OpenSSL dev headers, a C++17
# compiler, go-task (`task`) and `pixi`.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sdk="${root}/thirdparty/igris_c_sdk"

die() { echo "setup: $*" >&2; exit 1; }

if [ "${PIXI_PROJECT_ROOT:-}" = "${root}" ]; then
  die "run from a plain shell, not inside this repo's pixi env (pixi run / pixi shell)"
fi
for tool in git task pixi; do
  command -v "${tool}" >/dev/null 2>&1 || die "'${tool}' not found on PATH"
done
if [ -z "${AMENT_PREFIX_PATH:-}" ]; then
  die "ROS 2 not sourced (AMENT_PREFIX_PATH unset); e.g. 'source /opt/ros/jazzy/setup.bash'"
fi

echo "== [1/5] submodules"
git -C "${root}" submodule update --init thirdparty/gmr thirdparty/vr_link
git -C "${root}" submodule update --init --recursive thirdparty/igris_c_sdk

echo "== [2/5] pixi install"
pixi install --manifest-path "${root}/pyproject.toml"

echo "== [3/5] igris_c_sdk wheel (task wheel, SDK pixi env)"
(cd "${sdk}" && task wheel)

echo "== [4/5] install wheel into the gmr pixi env"
pixi run --manifest-path "${root}/pyproject.toml" install-sdk

echo "== [5/5] check-env"
pixi run --manifest-path "${root}/pyproject.toml" check-env
