"""``teleop/retarget_frame`` publisher: whole-body motion reference and active state over CycloneDDS.

Carries the same frame contract as ``RetargetFrameRosPublisher`` (``/gmr/teleop/retarget_frame`` +
``/gmr/teleop/active``) on the ``igris_c_sdk`` ``Joy_`` message under the robot namespace, so DDS-only
consumers (such as ``hit-igris-c-integration`` ``sim_robot``) need no ROS 2 / ``rclpy`` stack:

* Frame sample:
  - ``header.frame_id == "igris_c.retarget_frame.v1:<body_0>,<body_1>,..."``
  - ``buttons == [active (0|1), seq, body_count, flags]``
    where ``flags`` bitmask marks optional vectors:
    bit 0 = ``keybody_pos_local`` (``3 * B``), bit 1 = ``root_pos`` (``3``),
    bit 2 = ``root_vel`` (``3``), bit 3 = ``root_angvel`` (``3``).
  - ``axes`` (finite float32/float64 values):
    ``[fps, *dof_pos(31), *dof_vel(31), *root_rot_wxyz(4), *keybody_rot_local_wxyz(4*B), ...optional]``
* Active-edge sample (published on startup, pause/resume transitions, and shutdown):
  - ``header.frame_id == "igris_c.retarget_active.v1"``
  - ``buttons == [active (0|1), seq]``, ``axes == []``
"""

from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import Any

import numpy as np

from .channel import BODY_JOINT_COUNT, DdsChannel, sdk_type

logger = logging.getLogger("igris_c_teleop.dds")

DEFAULT_RETARGET_TOPIC = "teleop/retarget_frame"
RETARGET_FRAME_TAG = "igris_c.retarget_frame.v1"
RETARGET_ACTIVE_TAG = "igris_c.retarget_active.v1"

FLAG_KEYBODY_POS_LOCAL = 1 << 0
FLAG_ROOT_POS = 1 << 1
FLAG_ROOT_VEL = 1 << 2
FLAG_ROOT_ANGVEL = 1 << 3


def _flat_finite(values: Any, expected: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape != (expected,):
        raise ValueError(f"{name} must have {expected} values, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _stamp_fields(stamp: float) -> tuple[int, int]:
    sec = int(stamp)
    nanosec = int(round((stamp - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    elif nanosec < 0:
        sec -= 1
        nanosec += 1_000_000_000
    return sec, nanosec


def encode_retarget_frame(
    frame: Mapping[str, Any], *, active: bool = True, seq: int = 0
) -> tuple[str, list[float], list[int]]:
    """Encode a built contract frame dict into ``(frame_id, axes, buttons)`` for ``sdk.Joy_``."""
    for key in ("dof_pos", "dof_vel", "root_rot", "link_body_list", "keybody_rot_local"):
        if key not in frame:
            raise ValueError(f"frame lacks required key {key!r}.")

    bodies = [str(name) for name in np.asarray(frame["link_body_list"]).reshape(-1)]
    if not bodies or any(not name or "," in name or ":" in name for name in bodies):
        raise ValueError(f"invalid link_body_list: {bodies!r}.")
    body_count = len(bodies)

    fps = float(frame.get("fps", 0.0) or 0.0)
    if not np.isfinite(fps):
        raise ValueError("fps must be finite.")
    dof_pos = _flat_finite(frame["dof_pos"], BODY_JOINT_COUNT, "dof_pos")
    dof_vel = _flat_finite(frame["dof_vel"], BODY_JOINT_COUNT, "dof_vel")
    root_rot = _flat_finite(frame["root_rot"], 4, "root_rot")
    rot_local = _flat_finite(frame["keybody_rot_local"], body_count * 4, "keybody_rot_local")

    chunks = [np.array([fps], dtype=np.float64), dof_pos, dof_vel, root_rot, rot_local]
    flags = 0
    for key, bit, width in (
        ("keybody_pos_local", FLAG_KEYBODY_POS_LOCAL, body_count * 3),
        ("root_pos", FLAG_ROOT_POS, 3),
        ("root_vel", FLAG_ROOT_VEL, 3),
        ("root_angvel", FLAG_ROOT_ANGVEL, 3),
    ):
        if frame.get(key) is not None:
            chunks.append(_flat_finite(frame[key], width, key))
            flags |= bit

    frame_id = f"{RETARGET_FRAME_TAG}:{','.join(bodies)}"
    axes = np.concatenate(chunks).tolist()
    buttons = [1 if active else 0, int(seq), body_count, int(flags)]
    return frame_id, axes, buttons


def encode_retarget_active(active: bool, *, seq: int = 0) -> tuple[str, list[float], list[int]]:
    """Encode an operator active-state transition into ``(frame_id, axes, buttons)`` for ``sdk.Joy_``."""
    return RETARGET_ACTIVE_TAG, [], [1 if active else 0, int(seq)]


class RetargetFrameDdsPublisher:
    """Publishes whole-body retarget frames and active state on ``<ns>/teleop/retarget_frame``."""

    def __init__(self, channel: DdsChannel, topic: str = DEFAULT_RETARGET_TOPIC) -> None:
        sdk = channel.sdk
        self._joy_type = sdk_type(sdk, "Joy")
        self._header_type = sdk_type(sdk, "Header")
        self._time_type = sdk_type(sdk, "Time")
        self.topic = str(topic)
        self.resolved_topic = channel.resolve(self.topic)
        self._publisher = sdk.JoyPublisher(self.topic, sdk.QosProfile.SensorData())
        if not self._publisher.init():
            raise SystemExit(f"igris_c_sdk JoyPublisher failed to initialize on {self.topic!r}.")
        self._seq = 0
        self.published = 0
        self.active_published = 0
        self.rejected = 0

    def _write(self, frame_id: str, axes: list[float], buttons: list[int], stamp: float) -> bool:
        sec, nanosec = _stamp_fields(stamp)
        when = self._time_type()
        when.sec(sec)
        when.nanosec(nanosec)
        header = self._header_type()
        header.stamp(when)
        header.frame_id(frame_id)
        message = self._joy_type()
        message.header(header)
        message.axes(axes)
        message.buttons(buttons)
        if self._publisher.write(message) is False:
            self.rejected += 1
            return False
        return True

    def publish(self, frame: Mapping[str, Any], *, active: bool = True, stamp: float) -> None:
        self._seq += 1
        frame_id, axes, buttons = encode_retarget_frame(frame, active=active, seq=self._seq)
        if self._write(frame_id, axes, buttons, stamp):
            self.published += 1
            if self.published == 1:
                logger.info("First DDS retarget frame written on %s (%d axes)", self.resolved_topic, len(axes))

    def publish_active(self, active: bool, *, stamp: float) -> None:
        frame_id, axes, buttons = encode_retarget_active(active, seq=self._seq)
        if self._write(frame_id, axes, buttons, stamp):
            self.active_published += 1

    def close(self) -> None:
        try:
            stop = getattr(self._publisher, "stop", None)
            if stop is not None:
                stop()
        except Exception as exc:
            logger.warning("igris_c_sdk JoyPublisher stop failed: %s", exc)


__all__ = [
    "DEFAULT_RETARGET_TOPIC",
    "FLAG_KEYBODY_POS_LOCAL",
    "FLAG_ROOT_ANGVEL",
    "FLAG_ROOT_POS",
    "FLAG_ROOT_VEL",
    "RETARGET_ACTIVE_TAG",
    "RETARGET_FRAME_TAG",
    "RetargetFrameDdsPublisher",
    "encode_retarget_active",
    "encode_retarget_frame",
]
