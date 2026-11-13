"""Operator frame anchor: a resume continues from where the operator stood and faced before.

While paused the operator may turn and walk. The first live record after the
resume then shows the whole skeleton at another heading and position in one
tick. The robot does not follow that (the inference node holds the heading
while the headset is not Active, and the horizontal position is not a policy
input), but the jump itself is harmful: the inference node rotates its whole
reference history by one current heading correction, so the frames received
before an instantaneous turn read as turned by the full angle until they leave
the history window (about 0.2 s for a 21-frame window). The headset used to
hide this by blending the skeleton at 45 deg/s; it sends the live pose now.

This module removes the jump at its source. It applies one rigid transform in
the horizontal plane (a rotation about the vertical axis and an xy shift; z is
never touched) to every joint of the GMR input, and on each Start/Resume edge
it re-fixes that transform so that the first live skeleton has the heading and
the foot position of the last skeleton that went into the retargeting (the
frozen one). It is an alignment fixed at that instant, not a ramp: nothing is
left to move afterwards, and posture differences (joints, torso tilt, height)
stay for the convergence gate.

Heading is the horizontal direction of the hip line, position the midpoint of
the two foot joints: both are plain joint positions, independent of the
per-joint orientation convention. The transform and the last placement
survive a skeleton loss, so a reconnect (even after a headset restart with a
new tracking origin) is aligned to the reference that was last published.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from vr_link.types import TeleopState

from .converge import ResumeEdge

logger = logging.getLogger("igris_c_teleop.shaping.anchor")

HumanData = dict[str, list[np.ndarray]]

DEFAULT_HEADING_JOINTS: tuple[str, str] = ("Left_Hip", "Right_Hip")
DEFAULT_POSITION_JOINTS: tuple[str, str] = ("Left_Foot", "Right_Foot")


def _wrap(angle: float) -> float:
    return math.remainder(angle, 2.0 * math.pi)


class OperatorFrameAnchor:
    def __init__(
        self,
        *,
        enabled: bool = True,
        heading_joints: tuple[str, str] = DEFAULT_HEADING_JOINTS,
        position_joints: tuple[str, str] = DEFAULT_POSITION_JOINTS,
        min_baseline_m: float = 0.05,
    ) -> None:
        self.enabled = bool(enabled)
        self.heading_joints = tuple(heading_joints)
        self.position_joints = tuple(position_joints)
        self.min_baseline_m = float(min_baseline_m)
        self.yaw = 0.0  # rotation about the vertical axis applied to the raw skeleton (rad)
        self.shift = np.zeros(2)  # xy shift applied after the rotation (m)
        self._edge = ResumeEdge()
        self._last_heading: float | None = None  # placement of the last skeleton returned
        self._last_center: np.ndarray | None = None
        self.realigned = 0

    def reset(self) -> None:
        """Skeleton lost: forget the headset state only, so a reconnect in RESUMING aligns to the last placement."""

        self._edge.reset()

    def _placement(self, human: HumanData) -> tuple[float, np.ndarray] | None:
        """``(heading, xy)`` of a skeleton, or None when the hip line is too short to give a direction."""

        try:
            left, right = (np.asarray(human[name][0], dtype=np.float64) for name in self.heading_joints)
            feet = [np.asarray(human[name][0], dtype=np.float64) for name in self.position_joints]
        except KeyError:
            return None
        line = right[:2] - left[:2]
        if not np.all(np.isfinite(line)) or float(np.linalg.norm(line)) < self.min_baseline_m:
            return None
        center = (feet[0][:2] + feet[1][:2]) / 2.0
        if not np.all(np.isfinite(center)):
            return None
        return math.atan2(float(line[1]), float(line[0])), center

    def _transform(self, human: HumanData) -> HumanData:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        cw, sw = math.cos(self.yaw / 2.0), math.sin(self.yaw / 2.0)
        out: HumanData = {}
        for name, (position, quat) in human.items():
            x, y, z = (float(v) for v in position)
            w, qx, qy, qz = (float(v) for v in quat)
            out[name] = [
                np.array([c * x - s * y + self.shift[0], s * x + c * y + self.shift[1], z]),
                # (cw, 0, 0, sw) * (w, qx, qy, qz): the world-frame rotation about z.
                np.array([cw * w - sw * qz, cw * qx - sw * qy, cw * qy + sw * qx, cw * qz + sw * w]),
            ]
        return out

    def _realign(self, raw: HumanData) -> None:
        placement = self._placement(raw)
        if placement is None:
            logger.warning("operator frame not re-anchored at this resume: the skeleton gives no heading (hip line too short)")
            return
        heading, center = placement
        old_yaw, old_shift = self.yaw, self.shift
        yaw = _wrap(self._last_heading - heading)
        c, s = math.cos(yaw), math.sin(yaw)
        shift = self._last_center - np.array([c * center[0] - s * center[1], s * center[0] + c * center[1]])
        co, so = math.cos(old_yaw), math.sin(old_yaw)
        before = np.array([co * center[0] - so * center[1], so * center[0] + co * center[1]]) + old_shift
        self.yaw, self.shift = yaw, shift
        self.realigned += 1
        logger.info(
            "operator frame re-anchored at resume: the operator turned %.1f deg and moved %.2f m while not streaming live",
            math.degrees(_wrap(old_yaw - yaw)),
            float(np.linalg.norm(before - self._last_center)),
        )

    def update(self, human: HumanData, *, teleop_state: TeleopState, resume_count: int = 0) -> HumanData:
        """The skeleton to retarget for this record (a new dict; ``human`` is not modified)."""

        edge = self._edge.update(teleop_state, resume_count)
        if not self.enabled:
            return human
        if edge and self._last_heading is not None:
            self._realign(human)
        out = self._transform(human)
        placement = self._placement(out)
        if placement is not None:
            self._last_heading, self._last_center = placement
        return out


__all__ = ["DEFAULT_HEADING_JOINTS", "DEFAULT_POSITION_JOINTS", "OperatorFrameAnchor"]
