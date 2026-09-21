"""Builders for VRInputState / VRS1 records as the gmr-adapted device delivers them."""

from __future__ import annotations

import math

import numpy as np
from general_motion_retargeting.vr_uplink_convention import to_xrobot_convention
from vr_link.tcp_device import build_vr_state_record
from vr_link.types import (
    BODY_JOINT_NAMES,
    HAND_KEYPOINT_COUNT,
    ControllerInput,
    HandTrackingState,
    TrackedPose,
    VRInputState,
)

IDENTITY = (1.0, 0.0, 0.0, 0.0)


def yaw_quat(angle: float) -> tuple[float, float, float, float]:
    return (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


def pitch_quat(angle: float) -> tuple[float, float, float, float]:
    return (math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0)


def make_state(
    *,
    head_quat=IDENTITY,
    head_valid: bool = True,
    spine3_raw_quat=IDENTITY,
    spine3_valid: bool = True,
    left: ControllerInput = ControllerInput(),
    right: ControllerInput = ControllerInput(),
    connected: bool = True,
    timestamp: float = 1.5,
    left_hand: HandTrackingState | None = None,
    right_hand: HandTrackingState | None = None,
) -> VRInputState:
    """Body joints carry the gmr convention correction, the head stays raw."""

    joints = []
    for name in BODY_JOINT_NAMES:
        raw = spine3_raw_quat if name == "Spine3" else IDENTITY
        valid = spine3_valid if name == "Spine3" else True
        joints.append(TrackedPose(position_m=(0.0, 0.0, 1.0), quat_wxyz=to_xrobot_convention(name, raw), valid=valid))
    return VRInputState(
        timestamp=timestamp,
        head=TrackedPose(position_m=(0.0, 0.0, 1.7), quat_wxyz=head_quat, valid=head_valid),
        left_input=left,
        right_input=right,
        body_joints=tuple(joints),
        connected=connected,
        record_toggle_count=3,
        left_hand=left_hand,
        right_hand=right_hand,
    )


def click(right: bool = False, left: bool = False, b: bool = False) -> VRInputState:
    return make_state(
        right=ControllerInput(button_thumbstick=1.0 if right else 0.0, button_second=1.0 if b else 0.0),
        left=ControllerInput(button_thumbstick=1.0 if left else 0.0),
    )


def make_hand(tracked: bool = True, z: float = 0.1, count: int = HAND_KEYPOINT_COUNT) -> HandTrackingState:
    keypoints = tuple(TrackedPose(position_m=(0.0, 0.0, z), valid=True) for _ in range(count))
    return HandTrackingState(tracked=tracked, keypoints=keypoints)


def curl_from_tip_z(keypoints: np.ndarray, calibration) -> np.ndarray:
    """Stands in for hand_keypoints_to_normalized: every slot = the tip-row z."""

    del calibration
    assert keypoints.shape == (HAND_KEYPOINT_COUNT, 3)
    return np.full(6, float(keypoints[5, 2]))


def vrs1_record(grip: float = 0.0) -> bytes:
    joints = [((0.0, 0.0, 1.0), IDENTITY) for _ in BODY_JOINT_NAMES]
    return build_vr_state_record(
        joints,
        valid_mask=(1 << (len(BODY_JOINT_NAMES) + 1)) - 1,
        timestamp_ns=7,
        head=((0.0, 0.0, 1.7), IDENTITY),
        right_input=ControllerInput(grip=grip),
    )
