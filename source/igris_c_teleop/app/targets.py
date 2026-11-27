"""Teleop targets and the transport that carries the motion reference to each of them.

The target names what the session drives. ``igris-c-teleop`` (app/teleop.py) picks the headset video
source from it and the producer (app/producer.py) picks the reference transport from it:

* ``isaac``: the simulator stand-in (hit-igris-c-integration ``sim_robot``) reads the retarget frame
  from CycloneDDS, ``<ns>/teleop/retarget_frame`` (dds/retarget_frame.py). No ROS 2 publisher is opened.
* every other target: the frame goes out over ROS 2, ``/gmr/teleop/retarget_frame``
  (ros/publisher.py), which is what the robot's inference node subscribes to.
"""

from __future__ import annotations

TARGETS: tuple[str, ...] = ("robot", "usb", "mujoco", "isaac", "shm")
DEFAULT_TARGET = "robot"
DDS_REFERENCE_TARGETS: frozenset[str] = frozenset({"isaac"})


def reference_over_dds(target: str) -> bool:
    """Whether ``target`` takes the motion reference from CycloneDDS instead of ROS 2."""

    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of {TARGETS}")
    return target in DDS_REFERENCE_TARGETS


__all__ = ["DDS_REFERENCE_TARGETS", "DEFAULT_TARGET", "TARGETS", "reference_over_dds"]
