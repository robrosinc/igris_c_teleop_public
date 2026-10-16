#!/usr/bin/env bash
# Fresh-clone setup for igris_c_teleop, end to end:
#   1. submodules: gmr, vr_link (vr_link 2.1 carries no SDK of its own)
#   2. pixi install          (RoboStack ROS 2 + GMR deps + editable vr_link + the igris_c_sdk
#                             release wheel, a pypi URL dependency pinned by tag in pyproject.toml)
#   3. pixi run check-env    (SDK import path under .pixi/envs, not shadowed)
#
# The igris_c_sdk wheel is downloaded by pixi from github.com/robrosinc/igris_c_sdk_public
# (URL and version in pixi.lock, no content hash). No SDK source checkout, CMake, go-task or host ROS 2 is
# needed. Until 2026-10-16 this script also built that wheel from a thirdparty/igris_c_sdk
# submodule (`task wheel` + `pixi run install-sdk`).
#
# Host prerequisites (not installed here): git, pixi, network access to GitHub and to the
# conda / PyPI channels.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() { echo "setup: $*" >&2; exit 1; }

for tool in git pixi; do
  command -v "${tool}" >/dev/null 2>&1 || die "'${tool}' not found on PATH"
done

echo "== [1/3] submodules"
git -C "${root}" submodule update --init thirdparty/gmr thirdparty/vr_link

echo "== [2/3] pixi install"
pixi install --manifest-path "${root}/pyproject.toml"

echo "== [3/3] check-env"
pixi run --manifest-path "${root}/pyproject.toml" check-env
