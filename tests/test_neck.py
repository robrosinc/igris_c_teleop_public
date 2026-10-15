"""HMD-relative neck override: formula, MJCF clamp, qpos slots."""

from __future__ import annotations

import math

import numpy as np
import pytest
from vr_link.types import VRInputState

from igris_c_teleop import neck
from vr_fixtures import make_state, pitch_quat, yaw_quat


def test_neck_angles_head_yaw_left_is_positive():
    yaw, pitch = neck.head_relative_neck_angles(make_state(head_quat=yaw_quat(math.radians(30.0))))
    assert yaw == pytest.approx(math.radians(30.0), abs=1e-9)
    assert pitch == pytest.approx(0.0, abs=1e-9)


def test_neck_angles_head_pitch_down_is_positive():
    yaw, pitch = neck.head_relative_neck_angles(make_state(head_quat=pitch_quat(math.radians(20.0))))
    assert yaw == pytest.approx(0.0, abs=1e-9)
    assert pitch == pytest.approx(math.radians(20.0), abs=1e-9)


def test_neck_angles_are_chest_relative():
    turned = yaw_quat(math.radians(30.0))  # chest and head both yawed: GMR already turns the body
    yaw, pitch = neck.head_relative_neck_angles(make_state(head_quat=turned, spine3_raw_quat=turned))
    assert yaw == pytest.approx(0.0, abs=1e-9)
    assert pitch == pytest.approx(0.0, abs=1e-9)


def test_neck_angles_none_on_unusable_data():
    assert neck.head_relative_neck_angles(make_state(head_valid=False)) is None
    assert neck.head_relative_neck_angles(make_state(spine3_valid=False)) is None
    assert neck.head_relative_neck_angles(VRInputState(connected=True)) is None


def test_neck_override_targets_neck_qpos_slots(model, keyframe_qpos):
    override = neck.NeckOverride(model)
    assert override.qpos_addresses == (model.nq - 2, model.nq - 1) == (36, 37)
    qpos = keyframe_qpos.copy()
    written = override.apply(qpos, (0.2, -0.1))
    assert written == pytest.approx((0.2, -0.1))
    assert qpos[36] == pytest.approx(0.2) and qpos[37] == pytest.approx(-0.1)
    np.testing.assert_array_equal(qpos[:36], keyframe_qpos[:36])


def test_neck_override_clamps_to_mjcf_range(model, keyframe_qpos):
    override = neck.NeckOverride(model)
    (yaw_low, yaw_high), (pitch_low, pitch_high) = override.ranges
    assert yaw_low < 0.0 < yaw_high and pitch_low < 0.0 < pitch_high
    qpos = keyframe_qpos.copy()
    written = override.apply(qpos, (10.0, -10.0))
    assert written == pytest.approx((yaw_high, pitch_low))
    assert qpos[36] == pytest.approx(yaw_high) and qpos[37] == pytest.approx(pitch_low)


def test_neck_override_none_keeps_gmr_neck_and_signs_flip(model, keyframe_qpos):
    qpos = keyframe_qpos.copy()
    assert neck.NeckOverride(model).apply(qpos, None) is None
    np.testing.assert_array_equal(qpos, keyframe_qpos)
    flipped = neck.NeckOverride(model, yaw_sign=-1.0, pitch_sign=-1.0)
    assert flipped.apply(qpos, (0.2, 0.1)) == pytest.approx((-0.2, -0.1))


def test_neck_override_ends_up_in_dof_pos(model, keyframe_qpos):
    from igris_c_teleop.retarget_frame import RetargetFramePayloadBuilder

    qpos = keyframe_qpos.copy()
    neck.NeckOverride(model).apply(qpos, neck.head_relative_neck_angles(make_state(head_quat=yaw_quat(0.3))))
    frame = RetargetFramePayloadBuilder(model, robot_name="robros_igris_c_v2", target_fps=100.0).build(qpos, now=0.0)
    assert frame["dof_pos"][0][-2:] == pytest.approx([0.3, 0.0])
