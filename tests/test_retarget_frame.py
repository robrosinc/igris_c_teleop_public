"""Contract frame checks: shapes, whitelist, velocities, serialization, and the C++ scanner oracle."""

from __future__ import annotations

import json

import numpy as np
import pytest

import consumer_oracle as oracle
from igris_c_gmr import retarget_frame as rf

ROBOT = "robros_igris_c_v2"
EXPECTED_DOF = 31


def _builder(model, **overrides):
    kwargs = dict(robot_name=ROBOT, target_fps=100.0, velocity_lpf_hz=None, zero_velocity_frames=1)
    kwargs.update(overrides)
    return rf.RetargetFramePayloadBuilder(model, **kwargs)


def test_model_layout(model):
    assert model.nq == rf.FREE_JOINT_QPOS_SIZE + EXPECTED_DOF


def test_payload_schema_and_shapes(model, keyframe_qpos):
    frame = _builder(model).build(keyframe_qpos, now=0.0)

    assert tuple(frame) == rf.CONTRACT_KEYS  # whitelist, in order; no world-frame or side-channel keys
    assert np.asarray(frame["root_pos"]).shape == (1, 3)
    root_rot = np.asarray(frame["root_rot"])
    assert root_rot.shape == (1, 4)
    assert np.isclose(np.linalg.norm(root_rot[0]), 1.0)
    assert np.asarray(frame["root_vel"]).shape == (1, 3)
    assert np.asarray(frame["root_angvel"]).shape == (1, 3)
    assert np.asarray(frame["dof_pos"]).shape == (1, EXPECTED_DOF)
    assert np.asarray(frame["dof_vel"]).shape == (1, EXPECTED_DOF)
    body_names = frame["link_body_list"]
    assert np.asarray(frame["keybody_pos_local"]).shape == (1, len(body_names), 3)
    assert np.asarray(frame["keybody_rot_local"]).shape == (1, len(body_names), 4)
    assert len(set(body_names)) == len(body_names)
    assert isinstance(frame["record_meta"]["unix_ms"], int)
    assert frame["record_meta"]["frame_idx"] == 0
    assert frame["record_meta"]["source"] == rf.DEFAULT_SOURCE_TAG
    assert frame["fps"] == 100.0

    np.testing.assert_allclose(frame["dof_pos"][0], keyframe_qpos[7:])
    np.testing.assert_allclose(frame["root_pos"][0], keyframe_qpos[:3])
    np.testing.assert_allclose(frame["root_rot"][0], keyframe_qpos[3:7])


def test_required_gmt_bodies_present(model, keyframe_qpos):
    frame = _builder(model).build(keyframe_qpos, now=0.0)
    assert not set(rf.GMT_REQUIRED_BODY_NAMES) - set(frame["link_body_list"])
    assert "base_link" not in frame["link_body_list"]  # inserted by the receiver


def test_keybody_pos_local_is_root_relative(model, keyframe_qpos):
    builder = _builder(model)
    frame_a = builder.build(keyframe_qpos, now=0.0)
    shifted = keyframe_qpos.copy()
    shifted[0] += 1.0
    frame_b = builder.build(shifted, now=1.0)
    np.testing.assert_allclose(frame_a["keybody_pos_local"], frame_b["keybody_pos_local"], atol=1e-9)


def test_zero_velocity_window_then_finite_difference(model, keyframe_qpos):
    builder = _builder(model, zero_velocity_frames=2)
    q1 = keyframe_qpos.copy()
    q1[7:] += 0.05
    q1[0] += 0.02
    first = builder.build(keyframe_qpos, now=0.00)
    second = builder.build(keyframe_qpos, now=0.01)
    third = builder.build(q1, now=0.02)
    assert not np.any(np.asarray(first["dof_vel"]))
    assert not np.any(np.asarray(second["dof_vel"]))
    np.testing.assert_allclose(third["dof_vel"][0], (q1[7:] - keyframe_qpos[7:]) / 0.01, rtol=1e-6)
    np.testing.assert_allclose(third["root_vel"][0], [0.02 / 0.01, 0.0, 0.0], atol=1e-9)


def test_reset_restarts_zero_velocity_window(model, keyframe_qpos):
    builder = _builder(model, zero_velocity_frames=1)
    q1 = keyframe_qpos.copy()
    q1[7:] += 0.05
    builder.build(keyframe_qpos, now=0.00)
    assert np.any(np.asarray(builder.build(q1, now=0.01)["dof_vel"]))
    builder.reset()
    assert not np.any(np.asarray(builder.build(keyframe_qpos, now=0.02)["dof_vel"]))
    assert builder.frame_idx == 3  # frame index keeps counting across resets


def test_velocity_lpf_attenuates_step(model, keyframe_qpos):
    raw = _builder(model, velocity_lpf_hz=None)
    filtered = _builder(model, velocity_lpf_hz=8.0)
    q1 = keyframe_qpos.copy()
    q1[7:] += 0.05
    for builder in (raw, filtered):
        builder.build(keyframe_qpos, now=0.00)
    raw_vel = np.asarray(raw.build(q1, now=0.01)["dof_vel"][0])
    lpf_vel = np.asarray(filtered.build(q1, now=0.01)["dof_vel"][0])
    dt, tau = 0.01, 1.0 / (2.0 * np.pi * 8.0)
    alpha = dt / (tau + dt)
    np.testing.assert_allclose(lpf_vel, alpha * raw_vel, rtol=1e-6)
    assert np.linalg.norm(lpf_vel) < np.linalg.norm(raw_vel)


def test_rejects_wrong_qpos_length(model, keyframe_qpos):
    with pytest.raises(ValueError):
        _builder(model).build(keyframe_qpos[:-1], now=0.0)


def test_missing_required_body_raises(model, monkeypatch):
    monkeypatch.setattr(rf, "GMT_REQUIRED_BODY_NAMES", rf.GMT_REQUIRED_BODY_NAMES + ("Link_Does_Not_Exist",))
    with pytest.raises(ValueError):
        _builder(model)


def test_build_has_no_extra_channel(model, keyframe_qpos):
    with pytest.raises(TypeError):
        _builder(model).build(keyframe_qpos, now=0.0, extra={"hand_motor_q": [0.0] * 12})  # type: ignore[call-arg]
    assert not hasattr(rf, "mimic_obs")


# -- serialization / consumer contract -------------------------------------------------


def test_to_ros_payload_is_compact_json(model, keyframe_qpos):
    frame = _builder(model).build(keyframe_qpos, now=0.0)
    text = rf.to_ros_payload(frame)
    assert " " not in text and "NaN" not in text
    assert json.loads(text)["record_meta"]["robot"] == ROBOT


def test_to_ros_payload_rejects_nan_and_missing_keys(model, keyframe_qpos):
    frame = _builder(model).build(keyframe_qpos, now=0.0)
    frame["dof_pos"][0][3] = float("nan")
    with pytest.raises(ValueError):
        rf.to_ros_payload(frame)
    frame = _builder(model).build(keyframe_qpos, now=0.0)
    del frame["keybody_rot_local"]
    with pytest.raises(ValueError, match="keybody_rot_local"):
        rf.to_ros_payload(frame)


def test_consumer_scanner_accepts_payload(model, keyframe_qpos):
    """What igris_c_inference's string scanner extracts from the wire payload."""

    frame = _builder(model).build(keyframe_qpos, now=0.0)
    parsed = oracle.ingest(rf.to_ros_payload(frame), EXPECTED_DOF)
    assert parsed is not None, "consumer would drop the frame"
    np.testing.assert_allclose(parsed["dof_pos"], keyframe_qpos[7:])
    assert len(parsed["dof_vel"]) == EXPECTED_DOF
    np.testing.assert_allclose(parsed["root_rot"], keyframe_qpos[3:7])
    assert parsed["link_body_list"] == frame["link_body_list"]
    bodies = len(frame["link_body_list"])
    assert len(parsed["keybody_rot_local"]) == bodies * 4
    assert len(parsed["keybody_pos_local"]) == bodies * 3
    assert len(parsed["root_pos"]) == 3 and len(parsed["root_vel"]) == 3 and len(parsed["root_angvel"]) == 3


def test_record_meta_strings_never_contain_contract_keys(model, keyframe_qpos):
    """The scanner finds keys by substring; a meta string spelling a key would hijack it."""

    frame = _builder(model).build(keyframe_qpos, now=0.0)
    meta_text = json.dumps(frame["record_meta"])
    for key in rf.REQUIRED_CONSUMER_KEYS + ("keybody_pos_local", "root_pos", "root_vel", "root_angvel"):
        assert key not in meta_text.replace('"' + key + '"', "")  # key names appear only as JSON keys elsewhere
        assert oracle.find_key(meta_text, key) < 0


def test_scanner_drops_short_dof_pos(model, keyframe_qpos):
    frame = _builder(model).build(keyframe_qpos, now=0.0)
    frame["dof_pos"] = [frame["dof_pos"][0][:-1]]
    assert oracle.ingest(json.dumps(frame), EXPECTED_DOF) is None
