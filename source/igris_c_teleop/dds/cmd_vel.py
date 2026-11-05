"""``cmd_vel`` publisher: the operator's walk velocity for the robot's Low Level Walkmode.

Walks the real robot (igris_c_bridge subscribes ``cmd_vel`` with the Default QoS profile, as the
igris_c_sdk ``twist_publisher`` example does). ROBROS hardware-change approval and validation are
required before running against it.
"""

from __future__ import annotations

import logging
import math

from ..uplink.walk import WalkVelocity
from .channel import DdsChannel, sdk_type

logger = logging.getLogger("igris_c_teleop.dds")

DEFAULT_CMD_VEL_TOPIC = "cmd_vel"
CMD_VEL_FRAME_ID = "base_link"  # body frame of the planar command


class CmdVelDdsPublisher:
    """``cmd_vel`` publisher: one ``TwistStamped`` per call (linear x/y, angular z; the rest 0)."""

    def __init__(self, channel: DdsChannel, topic: str = DEFAULT_CMD_VEL_TOPIC) -> None:
        sdk = channel.sdk
        self._twist_type = sdk_type(sdk, "TwistStamped")
        self._header_type = sdk_type(sdk, "Header")
        self._time_type = sdk_type(sdk, "Time")
        self.topic = str(topic)
        self.resolved_topic = channel.resolve(self.topic)
        self._publisher = sdk.TwistStampedPublisher(self.topic, sdk.QosProfile.Default())
        if not self._publisher.init():
            raise SystemExit(f"igris_c_sdk TwistStampedPublisher failed to initialize on {self.topic!r}.")
        self.published = 0
        self.rejected = 0

    def publish(self, velocity: WalkVelocity, stamp: float) -> None:
        """Write one command; ``stamp`` is ``time.time()`` seconds of the controller sample."""

        values = (float(velocity.vx), float(velocity.vy), float(velocity.yaw_rate))
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"Walk velocity must be finite, got {velocity!r}.")
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
        header.frame_id(CMD_VEL_FRAME_ID)
        message = self._twist_type()
        message.header(header)
        twist = message.twist()
        twist.linear().x(values[0])
        twist.linear().y(values[1])
        twist.linear().z(0.0)
        twist.angular().x(0.0)
        twist.angular().y(0.0)
        twist.angular().z(values[2])
        message.twist(twist)
        if self._publisher.write(message) is False:
            self.rejected += 1
            return
        self.published += 1
        if self.published == 1:
            logger.info("First cmd_vel written on %s: vx=%.3f vy=%.3f yaw_rate=%.3f", self.resolved_topic, *values)

    def close(self) -> None:
        try:
            stop = getattr(self._publisher, "stop", None)
            if stop is not None:
                stop()
        except Exception as exc:
            logger.warning("igris_c_sdk TwistStampedPublisher stop failed: %s", exc)


__all__ = ["CMD_VEL_FRAME_ID", "DEFAULT_CMD_VEL_TOPIC", "CmdVelDdsPublisher"]
