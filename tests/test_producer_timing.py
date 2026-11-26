"""Producer loop with source timing: one reference frame per headset record.

Runs ``producer.main`` end to end with a scripted device, a recording ROS
publisher and a stand-in for GMR (no IK, no network, no sleep).
"""

from __future__ import annotations

import dataclasses
import json
import signal

import numpy as np
import pytest
from vr_link.types import VRInputState

import consumer_oracle as oracle
from igris_c_teleop.app import producer
from vr_fixtures import make_state

STEP = 0.01  # joint increment per retargeted record (rad)
MS = 1_000_000
ARGS = ["--no_dds", "--no_viewer", "--replay_vrs1", "scripted", "--arm_hold", "off"]
ARGS += ["--velocity_lpf_hz", "0", "--zero_velocity_frames", "1"]


class _ScriptedDevice:
    """Hands out one prepared state per poll, then asks the producer to stop."""

    def __init__(self, states) -> None:
        self._states = list(states)
        self._index = 0

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self._states)

    def advance(self, dt: float) -> VRInputState:
        del dt
        if self._index >= len(self._states):
            signal.raise_signal(signal.SIGINT)
            return VRInputState(connected=False)
        state = self._states[self._index]
        self._index += 1
        return state


class _RecordingRos:
    def __init__(self, **kwargs) -> None:
        del kwargs
        self.frames: list[dict] = []
        self.payloads: list[str] = []
        self.published = 0

    def publish(self, payload: str) -> None:
        self.payloads.append(payload)
        self.frames.append(json.loads(payload))
        self.published += 1

    def publish_active(self, active: bool) -> None:
        del active

    def subscription_count(self) -> int:
        return 0

    def close(self) -> None:
        pass


class _SteppingGmr:
    """Every retarget call moves all hinges by STEP, so a frame's velocity tells which time step it was divided by."""

    def __init__(self, model, qpos) -> None:
        self.model = model
        self.scaled_human_data = {}
        self.calls = 0
        self._qpos = qpos

    def retarget(self, human, offset_to_ground=False):
        del human, offset_to_ground
        qpos = self._qpos.copy()
        qpos[7:] += STEP * self.calls
        self.calls += 1
        return qpos


class _NoSleep:
    def __init__(self, **kwargs) -> None:
        del kwargs

    def sleep(self) -> None:
        pass


def _record(stamp_ns: int) -> VRInputState:
    return dataclasses.replace(make_state(), source_timestamp_ns=stamp_ns)


def _run(monkeypatch, model, keyframe_qpos, states, extra_args=()):
    import general_motion_retargeting
    import loop_rate_limiters

    gmr = _SteppingGmr(model, keyframe_qpos)
    ros = _RecordingRos()
    monkeypatch.setattr(general_motion_retargeting, "GeneralMotionRetargeting", lambda **kwargs: gmr)
    monkeypatch.setattr(loop_rate_limiters, "RateLimiter", _NoSleep)
    monkeypatch.setattr(producer, "RetargetFrameRosPublisher", lambda **kwargs: ros)
    monkeypatch.setattr(producer, "Vrs1FileSource", lambda path: _ScriptedDevice(states))
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        assert producer.main([*ARGS, *extra_args]) == 0
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return gmr, ros


def _meta(ros: _RecordingRos, key: str) -> list:
    return [frame["record_meta"].get(key) for frame in ros.frames]


def test_one_frame_per_headset_record(monkeypatch, model, keyframe_qpos):
    # 100 Hz polling of an irregular ~52 Hz sender: gaps of 19, 14 and 25 ms, most records polled twice.
    stamps = [1000 * MS, 1019 * MS, 1033 * MS, 1058 * MS]
    polls = [stamps[0], stamps[0], stamps[1], stamps[1], stamps[1], stamps[2], stamps[3], stamps[3]]
    gmr, ros = _run(monkeypatch, model, keyframe_qpos, [_record(stamp) for stamp in polls])

    assert gmr.calls == len(stamps)  # a repeated record is not retargeted again
    assert _meta(ros, "source_timestamp_ns") == stamps  # and each record is published exactly once
    assert _meta(ros, "frame_idx") == [0, 1, 2, 3]
    session_ids = set(_meta(ros, "session_id"))
    assert len(session_ids) == 1 and len(next(iter(session_ids))) == 32
    assert set(_meta(ros, "source_session_id")) == {f"{next(iter(session_ids))}:1"}

    # Velocity = joint step / headset time step, whatever the loop period was.
    for frame, dt_ms in zip(ros.frames[1:], (19, 14, 25)):
        assert np.max(np.abs(frame["dof_vel"][0])) == pytest.approx(STEP / (dt_ms * 1e-3), rel=1e-9)
    assert not np.any(ros.frames[0]["dof_vel"])

    # The consumer accepts every frame as new input on one timeline.
    consumer_stamps = [oracle.source_stamp(payload) for payload in ros.payloads]
    for previous, current in zip(consumer_stamps, consumer_stamps[1:]):
        assert oracle.stamp_accepts(previous, current) and not oracle.stamp_timeline_changed(previous, current)
    assert all(oracle.ingest(payload, model.nq - 7) is not None for payload in ros.payloads)


def test_reconnection_and_backward_clock_open_a_new_source_session(monkeypatch, model, keyframe_qpos):
    lost = VRInputState(connected=False)
    states = [_record(1000 * MS), _record(1020 * MS), lost, _record(1020 * MS), _record(1040 * MS)]
    states += [_record(5 * MS), _record(25 * MS)]
    gmr, ros = _run(monkeypatch, model, keyframe_qpos, states)

    assert gmr.calls == 6  # the stamp repeated across the reconnection is a new record
    segments = [value.rsplit(":", 1)[1] for value in _meta(ros, "source_session_id")]
    assert segments == ["1", "1", "2", "2", "3", "3"]  # reconnection, then a headset clock that went backwards
    assert _meta(ros, "frame_idx") == list(range(6))  # the frame counter never restarts within a run

    consumer_stamps = [oracle.source_stamp(payload) for payload in ros.payloads]
    changed = [oracle.stamp_timeline_changed(a, b) for a, b in zip(consumer_stamps, consumer_stamps[1:])]
    assert changed == [False, True, False, True, False]
    assert all(oracle.stamp_accepts(a, b) for a, b in zip(consumer_stamps, consumer_stamps[1:]))
    # No velocity is differenced across a change of the headset clock.
    assert not np.any(ros.frames[2]["dof_vel"]) and not np.any(ros.frames[4]["dof_vel"])
    assert np.max(np.abs(ros.frames[5]["dof_vel"][0])) == pytest.approx(STEP / 0.020, rel=1e-9)


def test_no_source_timing_restores_the_per_tick_behaviour(monkeypatch, model, keyframe_qpos):
    polls = [1000 * MS, 1000 * MS, 1019 * MS, 1019 * MS]
    gmr, ros = _run(monkeypatch, model, keyframe_qpos, [_record(stamp) for stamp in polls], ["--no_source_timing"])

    assert gmr.calls == len(polls) and len(ros.frames) == len(polls)  # every tick, repeated records included
    for frame in ros.frames:
        assert set(frame["record_meta"]) == {"source", "robot", "frame_idx", "unix_ms", "fps"}
    assert oracle.source_stamp(ros.payloads[0])["session"] == oracle.LEGACY_SESSION


def test_sender_without_timestamps_keeps_the_per_tick_behaviour(monkeypatch, model, keyframe_qpos):
    states = [make_state() for _ in range(3)]  # source_timestamp_ns 0: a sender older than the field
    gmr, ros = _run(monkeypatch, model, keyframe_qpos, states)

    assert gmr.calls == 3 and len(ros.frames) == 3
    for frame in ros.frames:
        assert "source_timestamp_ns" not in frame["record_meta"] and "source_session_id" not in frame["record_meta"]
        assert len(frame["record_meta"]["session_id"]) == 32  # the consumer still spots a producer restart by it


def test_source_timing_flag_default():
    assert not producer.parse_args([]).no_source_timing
    assert producer.parse_args(["--no_source_timing"]).no_source_timing
