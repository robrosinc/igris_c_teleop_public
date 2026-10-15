"""``gmr/refstate`` publisher (opt-in): the 31 retargeted joints for the headset overlay.

Display-only side channel for the vr_system headset; the robot bridge does not subscribe.
"""

from __future__ import annotations

import logging

import numpy as np

from .channel import BODY_JOINT_COUNT, DdsChannel, sdk_type

logger = logging.getLogger("igris_c_teleop.dds")

# Bare topic of the retargeted-joint reference stream read by the vr_system headset
# (ARobotTeleopPawn.ReferenceTopic). Deliberately not "masterarmstate": that name belongs to
# the master-arm device, and the robot bridge must not mistake this display stream for one.
DEFAULT_REF_TOPIC = "gmr/refstate"
REF_JOINT_COUNT = BODY_JOINT_COUNT


class RefStateDdsPublisher:
    """``gmr/refstate`` publisher: the 31 retargeted hinge joints as a ``MasterArmState`` message.

    Display-only side channel for the vr_system headset (hand-target overlay,
    vr_system/docs/Arm_Impedance_Overlay_Plan.md): the headset runs the arm FK on
    ``joint_state[i].q`` to draw where the operator's hand is on the robot and compares it
    with LowState. The type is reused so no IDL/SDK change is needed; only ``header.stamp``
    (producer wall clock, the same clock the video frames are stamped with) and
    ``joint_state[].q`` are filled, every other field stays at its default. The robot bridge
    does not subscribe to this topic.
    """

    def __init__(self, channel: DdsChannel, topic: str = DEFAULT_REF_TOPIC) -> None:
        sdk = channel.sdk
        self._state_type = sdk_type(sdk, "MasterArmState")
        self._joint_type = sdk_type(sdk, "JointState")
        self._header_type = sdk_type(sdk, "Header")
        self._time_type = sdk_type(sdk, "Time")
        self.topic = str(topic)
        self.resolved_topic = channel.resolve(self.topic)
        self._publisher = sdk.MasterArmStatePublisher(self.topic, sdk.QosProfile.SensorData())
        if not self._publisher.init():
            raise SystemExit(f"igris_c_sdk MasterArmStatePublisher failed to initialize on {self.topic!r}.")
        self.published = 0
        self.rejected = 0

    def publish(self, joint_positions, stamp: float) -> None:
        """Write one reference sample; ``stamp`` is ``time.time()`` seconds of the solve."""

        values = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
        if values.shape != (REF_JOINT_COUNT,):
            raise ValueError(f"Reference joints must have {REF_JOINT_COUNT} values, got shape {values.shape}.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Reference joints contain NaN/inf.")
        joints = []
        for value in values:
            joint = self._joint_type()
            joint.q(float(value))
            joints.append(joint)
        sec = int(stamp)
        nanosec = int(round((stamp - sec) * 1e9))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        when = self._time_type()
        when.sec(sec)
        when.nanosec(nanosec)
        header = self._header_type()
        header.stamp(when)
        message = self._state_type()
        message.header(header)
        message.joint_state(joints)
        if self._publisher.write(message) is False:
            self.rejected += 1
            return
        self.published += 1
        if self.published == 1:
            logger.info("First RefState written on %s (%d joints, stamp %d.%09d)", self.resolved_topic, REF_JOINT_COUNT, sec, nanosec)

    def close(self) -> None:
        try:
            stop = getattr(self._publisher, "stop", None)
            if stop is not None:
                stop()
        except Exception as exc:
            logger.warning("igris_c_sdk MasterArmStatePublisher stop failed: %s", exc)


__all__ = ["DEFAULT_REF_TOPIC", "REF_JOINT_COUNT", "RefStateDdsPublisher"]
