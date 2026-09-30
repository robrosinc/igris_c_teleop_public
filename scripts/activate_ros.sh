# pixi activation hook (sourced, not executed).
#
# ROS 2 comes from RoboStack packages inside this env; their own activation scripts set
# AMENT_PREFIX_PATH, ROS_DISTRO, etc. The repo package and vr_link are installed
# editable. The only thing left to expose by hand is the vendored gmr submodule, whose
# setup.py pulls smplx/opencv/redis the producer does not need.
#
# PYTHONPATH is written from scratch on purpose: the login shell layers
# ~/workspace/repos/igris_c/install, whose rosidl package is also named `igris_c_sdk` and
# would shadow the pybind wheel. LD_LIBRARY_PATH is not touched (conda libraries resolve
# through rpath) and nothing from /opt/ros is referenced.

_root="${PIXI_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export PYTHONPATH="${_root}/thirdparty/gmr"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

unset _root
