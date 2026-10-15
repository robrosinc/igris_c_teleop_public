"""Hold a robot arm at its last reference pose while the operator's bare hand is untracked.

Bare-hand teleoperation (no controller, no wrist tracker) gives the PICO body
tracker nothing to anchor the arm on once the hand leaves the headset's view:
the wrist/elbow/shoulder body joints it keeps sending are not trustworthy and
GMR would move the robot arm with them. The finger command already holds in
that situation (``HandSourceArbiter`` -> ``hold``); this module gives the arm
the same policy, in joint space after GMR: the 7 arm joints of the retargeted
``qpos`` are frozen at the values of the tick the hold engaged, so the arm
keeps its configuration relative to the torso. The same hold engages when any
body joint of the arm chain (Shoulder, Elbow, Wrist) arrives with
``valid=False``: the sender puts an identity pose at the origin in that slot
and GMR would otherwise pull the arm there.

Release: once the hand is tracked again for ``release_stable_s`` (flicker
guard) the held pose is ramped to the live reference with a
:class:`~.ramp.JointRamp` (rate-capped) instead of jumping. Engaging is
immediate. Per side, independent.
"""

from __future__ import annotations

import logging

import numpy as np
from vr_link.types import BODY_JOINT_NAMES, VRInputState

from .ramp import JointRamp

logger = logging.getLogger("igris_c_teleop.arm_hold")

SIDES = ("left", "right")
_SIDE_SUFFIX = {"left": "Left", "right": "Right"}
# MJCF joint names of one arm (robros_igris_c_v2), shoulder to wrist.
ARM_JOINT_TEMPLATES = (
    "Joint_Shoulder_Pitch_{side}",
    "Joint_Shoulder_Roll_{side}",
    "Joint_Shoulder_Yaw_{side}",
    "Joint_Elbow_Pitch_{side}",
    "Joint_Wrist_Yaw_{side}",
    "Joint_Wrist_Roll_{side}",
    "Joint_Wrist_Pitch_{side}",
)
# VRS1 body joints GMR's xrobot IK config reads for one arm.
ARM_BODY_JOINTS = {
    "left": ("Left_Shoulder", "Left_Elbow", "Left_Wrist"),
    "right": ("Right_Shoulder", "Right_Elbow", "Right_Wrist"),
}


def arm_qpos_addresses(model, side: str) -> np.ndarray:
    """qpos addresses of the 7 arm joints of ``side`` in the GMR MuJoCo model."""

    import mujoco as mj

    addresses = []
    for template in ARM_JOINT_TEMPLATES:
        name = template.format(side=_SIDE_SUFFIX[side])
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint {name!r} not found in the GMR robot model.")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(addresses, dtype=np.int64)


def arm_chain_valid(state: VRInputState, side: str) -> bool:
    """True when every body joint of the arm chain carries a valid pose."""

    if len(state.body_joints) != len(BODY_JOINT_NAMES):
        return False
    return all(state.body_joints[BODY_JOINT_NAMES.index(name)].valid for name in ARM_BODY_JOINTS[side])


class _SideHold:
    def __init__(self, indices: np.ndarray) -> None:
        self.indices = indices
        self.holding = False
        self.held: np.ndarray | None = None
        self.tracked_since: float | None = None
        self.ramp: JointRamp | None = None
        self.reason = ""

    def reset(self) -> None:
        self.holding = False
        self.held = None
        self.tracked_since = None
        self.ramp = None
        self.reason = ""


class ArmHold:
    def __init__(
        self,
        model,
        *,
        enabled: bool = True,
        release_stable_s: float = 0.2,
        release_rate: float = 0.5,
        release_duration_s: float = 1.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.release_stable_s = float(release_stable_s)
        self.release_rate = float(release_rate)
        self.release_duration_s = float(release_duration_s)
        self._sides = {side: _SideHold(arm_qpos_addresses(model, side)) for side in SIDES}
        self.engaged = 0
        self.released = 0

    @property
    def holding(self) -> tuple[bool, bool]:
        return self._sides["left"].holding, self._sides["right"].holding

    @property
    def releasing(self) -> tuple[bool, bool]:
        return self._sides["left"].ramp is not None, self._sides["right"].ramp is not None

    def indices(self, side: str) -> np.ndarray:
        return self._sides[side].indices

    def reset(self) -> None:
        """Skeleton lost: forget holds and release ramps (the next connection starts clean)."""

        for hold in self._sides.values():
            hold.reset()

    @staticmethod
    def hold_requested(state: VRInputState, side: str, hand_source: str) -> tuple[bool, str]:
        """(requested, reason): finger arbiter says ``hold`` (bare hand untracked) or the arm chain is invalid."""

        if not arm_chain_valid(state, side):
            return True, "arm body joints invalid"
        if hand_source == "hold":
            return True, "hand tracking lost"
        return False, ""

    def update(self, qpos: np.ndarray, *, now: float, requests: dict[str, tuple[bool, str]]) -> tuple[bool, bool]:
        """Shape the arm joints of ``qpos`` in place; return (left_holding, right_holding)."""

        if not self.enabled:
            return False, False
        for side in SIDES:
            hold = self._sides[side]
            requested, reason = requests.get(side, (False, ""))
            if requested:
                if not hold.holding:
                    # Freeze what the robot is currently told (the release ramp output if one is running).
                    hold.held = hold.ramp.output if hold.ramp is not None else np.array(qpos[hold.indices], dtype=np.float64)
                    hold.ramp = None
                    hold.holding = True
                    hold.reason = reason
                    self.engaged += 1
                    logger.warning("%s arm HELD at its last reference pose (%s)", side, reason)
                hold.tracked_since = None
                qpos[hold.indices] = hold.held
                continue
            if hold.holding:
                # Tracking is back: wait for it to be stable, then ramp out of the hold.
                if hold.tracked_since is None:
                    hold.tracked_since = now
                if now - hold.tracked_since < self.release_stable_s:
                    qpos[hold.indices] = hold.held
                    continue
                hold.ramp = JointRamp(
                    hold.indices, hold.held, duration_s=self.release_duration_s, max_rate=self.release_rate, now=now
                )
                hold.holding = False
                hold.held = None
                hold.tracked_since = None
                self.released += 1
                logger.warning("%s arm released: ramping back to the live reference (%.1f s, %.2f rad/s cap)", side, self.release_duration_s, self.release_rate)
            if hold.ramp is not None:
                hold.ramp.apply(qpos, now)
                if hold.ramp.done:
                    hold.ramp = None
        return self.holding


__all__ = ["ARM_BODY_JOINTS", "ARM_JOINT_TEMPLATES", "SIDES", "ArmHold", "arm_chain_valid", "arm_qpos_addresses"]
