"""Latest-value readers of the robot's ``lowstate`` and ``robotstate`` topics.

Inputs of the convergence ramp (``shaping.converge``): the measured joint positions the
ramp starts from and the control mode whose low-level entry triggers it. Read-only; nothing
here commands the robot.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from .channel import BODY_JOINT_COUNT, DdsChannel

logger = logging.getLogger("igris_c_teleop.dds")

DEFAULT_LOWSTATE_TOPIC = "lowstate"
DEFAULT_ROBOTSTATE_TOPIC = "robotstate"


class LowStateSubscriber:
    """``lowstate`` reader keeping the newest 31 joint positions with their receive age.

    Input of the convergence gate (the ramp starts from the robot's measured
    pose). Age is measured with the local monotonic clock at receive time, not
    the robot's header stamp, so clock offsets between NUC and PC do not matter.
    """

    def __init__(self, channel: DdsChannel, topic: str = DEFAULT_LOWSTATE_TOPIC, *, joint_count: int = BODY_JOINT_COUNT) -> None:
        sdk = channel.sdk
        self.topic = str(topic)
        self.resolved_topic = channel.resolve(self.topic)
        self.joint_count = int(joint_count)
        self._subscriber = sdk.LowStateSubscriber(self.topic, sdk.QosProfile.SensorData())
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._received: float = 0.0
        self.received = 0
        self.errors = 0
        if not self._subscriber.init(self._on_message):
            raise SystemExit(f"igris_c_sdk LowStateSubscriber failed to initialize on {self.resolved_topic!r}.")
        logger.info("Subscribed LowState on %s (convergence ramp input)", self.resolved_topic)

    def _on_message(self, message) -> None:
        # SDK thread: copy out, never raise.
        try:
            joints = message.joint_state()
            values = np.fromiter((joints[i].q() for i in range(self.joint_count)), dtype=np.float64, count=self.joint_count)
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            if self.errors in (1, 10, 100) or self.errors % 1000 == 0:
                logger.warning("LowState on %s: extractor failed (%d so far): %s", self.resolved_topic, self.errors, exc)
            return
        with self._lock:
            self._latest = values
            self._received = time.monotonic()
            self.received += 1

    def latest(self) -> tuple[np.ndarray, float] | None:
        """``(q, age_s)`` of the newest sample, or None before the first one."""

        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy(), time.monotonic() - self._received

    def close(self) -> None:
        try:
            stop = getattr(self._subscriber, "stop", None)
            if stop is not None:
                stop()
        except Exception as exc:
            logger.warning("igris_c_sdk LowStateSubscriber stop failed: %s", exc)


class RobotModeSubscriber:
    """``robotstate`` reader keeping the newest ``RobotState_.state()`` (control mode) with its age."""

    def __init__(self, channel: DdsChannel, topic: str = DEFAULT_ROBOTSTATE_TOPIC) -> None:
        sdk = channel.sdk
        self.topic = str(topic)
        self.resolved_topic = channel.resolve(self.topic)
        self._subscriber = sdk.RobotStateSubscriber(self.topic, sdk.QosProfile.Default())
        self._lock = threading.Lock()
        self._latest: int | None = None
        self._received: float = 0.0
        self.received = 0
        self.errors = 0
        if not self._subscriber.init(self._on_message):
            raise SystemExit(f"igris_c_sdk RobotStateSubscriber failed to initialize on {self.resolved_topic!r}.")
        logger.info("Subscribed RobotState on %s (low-level mode entry detection)", self.resolved_topic)

    def _on_message(self, message) -> None:
        try:
            mode = int(message.state())
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            if self.errors in (1, 10, 100) or self.errors % 1000 == 0:
                logger.warning("RobotState on %s: extractor failed (%d so far): %s", self.resolved_topic, self.errors, exc)
            return
        with self._lock:
            self._latest = mode
            self._received = time.monotonic()
            self.received += 1

    def latest(self) -> tuple[int, float] | None:
        """``(mode, age_s)`` of the newest sample, or None before the first one."""

        with self._lock:
            if self._latest is None:
                return None
            return self._latest, time.monotonic() - self._received

    def close(self) -> None:
        try:
            stop = getattr(self._subscriber, "stop", None)
            if stop is not None:
                stop()
        except Exception as exc:
            logger.warning("igris_c_sdk RobotStateSubscriber stop failed: %s", exc)


__all__ = ["DEFAULT_LOWSTATE_TOPIC", "DEFAULT_ROBOTSTATE_TOPIC", "LowStateSubscriber", "RobotModeSubscriber"]
