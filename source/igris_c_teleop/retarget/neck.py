"""HMD-relative neck override for the GMR qpos.

GMR's IK neck is replaced by the head orientation relative to the operator's
chest (Spine3): the body-yaw component is already applied to the robot by GMR,
so only the head-vs-chest rotation drives the neck.
"""

from __future__ import annotations

import math

import numpy as np
from general_motion_retargeting.vr_uplink_convention import JOINT_CORRECTIONS_WXYZ, quat_mul_wxyz
from vr_link.types import BODY_JOINT_NAMES, VRInputState

SPINE3_INDEX = BODY_JOINT_NAMES.index("Spine3")

# MJCF joint names of the igris_c_v2 neck (dof 29/30 = qpos 36/37).
NECK_JOINT_NAMES = ("Joint_Neck_Yaw", "Joint_Neck_Pitch")


def _quat_conj(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (q[0], -q[1], -q[2], -q[3])


def head_relative_neck_angles(state: VRInputState) -> tuple[float, float] | None:
    """Neck yaw/pitch (rad) of the HMD relative to the operator's chest (Spine3).

    ``state.head`` is raw sender frame while ``state.body_joints`` carry the
    gmr convention correction, hence the Spine3 correction is undone first.
    Positive yaw = looking left, positive pitch = looking down (ZYX
    extraction). ``None`` when head or skeleton data is unusable; the caller
    then keeps GMR's IK neck.
    """

    if not state.head.valid or len(state.body_joints) != len(BODY_JOINT_NAMES):
        return None
    spine = state.body_joints[SPINE3_INDEX]
    if not spine.valid:
        return None

    spine_raw = quat_mul_wxyz(spine.quat_wxyz, _quat_conj(JOINT_CORRECTIONS_WXYZ["Spine3"]))
    w, x, y, z = quat_mul_wxyz(_quat_conj(spine_raw), state.head.quat_wxyz)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return yaw, pitch


class NeckOverride:
    """Writes HMD-derived neck angles into the neck slots of a MuJoCo qpos.

    Slots and limits come from the MJCF (``jnt_qposadr``/``jnt_range``), so
    the override tracks the model instead of hard-coded indices. Angles are
    sign-adjusted then clamped to the joint range.
    """

    def __init__(self, model, *, yaw_sign: float = 1.0, pitch_sign: float = 1.0) -> None:
        import mujoco as mj

        addresses: list[int] = []
        ranges: list[tuple[float, float]] = []
        for name in NECK_JOINT_NAMES:
            joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Joint {name!r} not found in the GMR robot model.")
            addresses.append(int(model.jnt_qposadr[joint_id]))
            if model.jnt_limited[joint_id]:
                low, high = model.jnt_range[joint_id]
                ranges.append((float(low), float(high)))
            else:
                ranges.append((-math.inf, math.inf))
        self._addresses = tuple(addresses)
        self._ranges = tuple(ranges)
        self._signs = (float(yaw_sign), float(pitch_sign))

    @property
    def qpos_addresses(self) -> tuple[int, ...]:
        return self._addresses

    @property
    def ranges(self) -> tuple[tuple[float, float], ...]:
        return self._ranges

    def apply(self, qpos: np.ndarray, angles: tuple[float, float] | None) -> tuple[float, float] | None:
        """Overwrite the neck slots of ``qpos`` in place; return the written values.

        ``angles=None`` leaves GMR's IK neck untouched and returns None.
        """

        if angles is None:
            return None
        written: list[float] = []
        for address, (low, high), sign, angle in zip(self._addresses, self._ranges, self._signs, angles):
            value = min(high, max(low, sign * float(angle)))
            qpos[address] = value
            written.append(value)
        return written[0], written[1]


__all__ = ["NECK_JOINT_NAMES", "SPINE3_INDEX", "NeckOverride", "head_relative_neck_angles"]
