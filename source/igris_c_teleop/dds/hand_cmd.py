"""``handcmd`` publisher: the 12 normalized finger motor targets.

Drives the real robot's hands. ROBROS hardware-change approval and validation are
required before running against it.
"""

from __future__ import annotations

import logging

import numpy as np

from ..retarget.hands import HAND_COMMAND_DIM, MOTOR_IDS
from .channel import DdsChannel, sdk_type

logger = logging.getLogger("igris_c_teleop.dds")

DEFAULT_HAND_TOPIC = "handcmd"


class HandCmdDdsPublisher:
    """``handcmd`` publisher: 12 normalized motor targets in ``MOTOR_IDS`` order."""

    def __init__(self, channel: DdsChannel, topic: str = DEFAULT_HAND_TOPIC) -> None:
        sdk = channel.sdk
        self._motor_cmd_type = sdk_type(sdk, "MotorCmd")
        self._hand_cmd_type = sdk_type(sdk, "HandCmd")
        self.topic = str(topic)
        self.resolved_topic = channel.resolve(self.topic)
        self._publisher = sdk.HandCmdPublisher(self.topic, sdk.QosProfile.SensorData())
        if not self._publisher.init():
            raise SystemExit(f"igris_c_sdk HandCmdPublisher failed to initialize on {self.topic!r}.")
        self.published = 0
        self.rejected = 0

    def publish(self, command) -> None:
        values = np.asarray(command, dtype=np.float64).reshape(-1)
        if values.shape != (HAND_COMMAND_DIM,):
            raise ValueError(f"Hand command must have {HAND_COMMAND_DIM} values, got shape {values.shape}.")
        motors = []
        for motor_id, value in zip(MOTOR_IDS, values):
            motor = self._motor_cmd_type()
            motor.id(int(motor_id))
            motor.q(float(min(1.0, max(0.0, value))))
            motors.append(motor)
        message = self._hand_cmd_type()
        message.motor_cmd(motors)
        if self._publisher.write(message) is False:
            self.rejected += 1
            return
        self.published += 1
        if self.published == 1:
            logger.info("First HandCmd written on %s: ids=%s q=%s", self.resolved_topic, list(MOTOR_IDS), np.round(values, 3).tolist())

    def close(self) -> None:
        try:
            stop = getattr(self._publisher, "stop", None)
            if stop is not None:
                stop()
        except Exception as exc:
            logger.warning("igris_c_sdk HandCmdPublisher stop failed: %s", exc)


__all__ = ["DEFAULT_HAND_TOPIC", "HandCmdDdsPublisher"]
