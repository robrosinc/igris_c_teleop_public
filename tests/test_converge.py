"""ConvergeGate: triggers (teleop_state, legacy connect, low-level entry), ramping, wait for LowState, resume echo."""

from __future__ import annotations

import numpy as np
import pytest
from vr_link.types import TeleopState

from igris_c_teleop.shaping.converge import TRIGGER_CONNECT, TRIGGER_LOW_LEVEL, TRIGGER_RESUME, ConvergeGate, low_level_modes_from_sdk

HINGES = np.arange(7, 38)
LOW, WALK, WALK_LOW = 3, 10, 6


def _gate(**kwargs):
    defaults = dict(duration_s=1.0, max_rate=0.0, hand_rate=1.0, lowstate_max_age_s=0.5)
    defaults.update(kwargs)
    return ConvergeGate(HINGES, **defaults)


def _qpos(value: float) -> np.ndarray:
    q = np.zeros(38)
    q[HINGES] = value
    return q


def _tick(gate, qpos, now, *, state=TeleopState.ACTIVE, edge=False, mode=None, lowstate=None, resume=0):
    return gate.update(qpos, now=now, teleop_state=state, connected_edge=edge, robot_mode=mode, lowstate=lowstate, resume_count=resume)


def test_resuming_starts_a_ramp_from_lowstate_and_converges():
    gate = _gate()
    lowstate = (np.full(31, 1.0), 0.01)
    status = _tick(gate, _qpos(2.0), 0.0, state=TeleopState.COUNTDOWN, lowstate=lowstate)
    assert not status.converging and status.trigger is None
    qpos = _qpos(2.0)
    status = _tick(gate, qpos, 1.0, state=TeleopState.RESUMING, lowstate=lowstate)
    assert status.converging and status.trigger == TRIGGER_RESUME and status.progress == 0.0
    assert np.allclose(qpos[HINGES], 1.0) and np.allclose(qpos[:7], 0.0)  # starts at the robot, root untouched
    qpos = _qpos(2.0)
    status = _tick(gate, qpos, 1.5, state=TeleopState.RESUMING, lowstate=lowstate)
    assert status.converging and np.allclose(qpos[HINGES], 1.5) and status.trigger is None
    qpos = _qpos(2.0)
    status = _tick(gate, qpos, 2.0, state=TeleopState.ACTIVE, lowstate=lowstate)
    assert not status.converging and status.progress == 1.0 and np.allclose(qpos[HINGES], 2.0)
    assert gate.started == 1 and gate.deferred == 0
    # Staying in RESUMING does not re-trigger; a new RESUMING after ACTIVE does.
    assert _tick(gate, _qpos(2.0), 3.0, state=TeleopState.RESUMING, lowstate=lowstate).trigger == TRIGGER_RESUME
    assert _tick(gate, _qpos(2.0), 3.1, state=TeleopState.RESUMING, lowstate=lowstate).trigger is None
    assert gate.started == 2


def test_ramp_releases_at_once_when_the_operator_is_already_at_the_robot():
    gate = _gate(duration_s=1.0, max_rate=0.5)
    qpos = _qpos(1.01)
    status = _tick(gate, qpos, 0.0, state=TeleopState.RESUMING, lowstate=(np.full(31, 1.0), 0.0))
    assert status.trigger == TRIGGER_RESUME and not status.converging and status.progress == 1.0
    assert gate.started == 1 and not gate.active and np.allclose(qpos[HINGES], 1.0)
    # The residual is reported while the ramp runs and the default blend time is 1 s.
    gate = ConvergeGate(HINGES, max_rate=0.0)
    assert gate.duration_s == 1.0
    status = _tick(gate, _qpos(2.0), 0.0, state=TeleopState.RESUMING, lowstate=(np.zeros(31), 0.0))
    assert status.converging and status.residual == 2.0
    status = _tick(gate, _qpos(2.0), 0.5, state=TeleopState.RESUMING, lowstate=(np.zeros(31), 0.0))
    assert status.converging and status.residual == 1.0 and status.progress == 0.5


def test_legacy_headset_triggers_on_connection_edge_only_when_state_is_unknown():
    gate = _gate()
    lowstate = (np.zeros(31), 0.0)
    assert _tick(gate, _qpos(1.0), 0.0, state=TeleopState.UNKNOWN, edge=True, lowstate=lowstate).trigger == TRIGGER_CONNECT
    gate.reset()
    assert _tick(gate, _qpos(1.0), 1.0, state=TeleopState.ACTIVE, edge=True, lowstate=lowstate).trigger is None
    gate.reset()
    # A reconnect straight into RESUMING counts as a resume, not as a legacy connect.
    assert _tick(gate, _qpos(1.0), 2.0, state=TeleopState.RESUMING, edge=True, lowstate=lowstate).trigger == TRIGGER_RESUME


def test_low_level_entry_triggers_but_starting_in_low_level_does_not():
    gate = _gate()
    lowstate = (np.zeros(31), 0.0)
    assert _tick(gate, _qpos(1.0), 0.0, mode=LOW, lowstate=lowstate).trigger is None  # first sample: already LOW
    assert _tick(gate, _qpos(1.0), 0.1, mode=WALK, lowstate=lowstate).trigger is None
    assert _tick(gate, _qpos(1.0), 0.2, mode=LOW, lowstate=lowstate).trigger == TRIGGER_LOW_LEVEL
    assert _tick(gate, _qpos(1.0), 0.3, mode=WALK_LOW, lowstate=lowstate).trigger is None  # within the family
    assert _tick(gate, _qpos(1.0), 0.4, mode=WALK, lowstate=lowstate).trigger is None
    assert _tick(gate, _qpos(1.0), 0.5, mode=WALK_LOW, lowstate=lowstate).trigger == TRIGGER_LOW_LEVEL
    # The robot mode memory survives a skeleton reset (the robot did not change).
    gate.reset()
    assert _tick(gate, _qpos(1.0), 0.6, mode=WALK_LOW, lowstate=lowstate).trigger is None


def test_first_start_without_fresh_lowstate_publishes_nothing_until_one_arrives():
    gate = _gate()
    qpos = _qpos(2.0)
    status = _tick(gate, qpos, 0.0, state=TeleopState.RESUMING, lowstate=None, resume=5)
    # No earlier reference to hold: the caller must not publish, and the request is not acknowledged.
    assert status.waiting and status.converging and not status.publish and status.deferred == TRIGGER_RESUME
    assert status.trigger is None and status.converged_resume_count == 0 and gate.waiting and not gate.active
    status = _tick(gate, _qpos(2.0), 0.1, state=TeleopState.RESUMING, lowstate=(np.zeros(31), 0.9), resume=5)  # too old
    assert status.waiting and not status.publish and status.deferred is None and gate.deferred == 1 and gate.started == 0
    # The first fresh LowState starts the ramp from the robot.
    qpos = _qpos(2.0)
    status = _tick(gate, qpos, 0.2, state=TeleopState.RESUMING, lowstate=(np.full(31, 1.0), 0.01), resume=5)
    assert status.trigger == TRIGGER_RESUME and status.converging and not status.waiting and status.publish
    assert np.allclose(qpos[HINGES], 1.0) and gate.started == 1 and not gate.waiting


def test_trigger_without_fresh_lowstate_holds_the_last_reference():
    gate = _gate()
    lowstate = (np.full(31, 1.0), 0.01)
    held = _qpos(1.0)
    held[:7] = [0.1, 0.2, 0.9, 1.0, 0.0, 0.0, 0.0]
    _tick(gate, held.copy(), 0.0, state=TeleopState.PAUSED, lowstate=lowstate)  # the frozen reference that went out
    # Resume while the robot state is stale: the whole reference (root included) stays where it was.
    live = _qpos(2.0)
    live[:7] = [5.0, 5.0, 1.2, 0.0, 1.0, 0.0, 0.0]
    qpos = live.copy()
    status = _tick(gate, qpos, 1.0, state=TeleopState.RESUMING, lowstate=(np.full(31, 1.0), 0.9), resume=7)
    assert status.waiting and status.publish and status.converging and status.deferred == TRIGGER_RESUME
    assert np.array_equal(qpos, held) and status.converged_resume_count == 0
    qpos = live.copy()
    status = _tick(gate, qpos, 1.5, state=TeleopState.RESUMING, lowstate=None, resume=7)
    assert status.waiting and np.array_equal(qpos, held) and gate.deferred == 1
    # A fresh LowState ends the wait: the ramp starts at the robot and the root is the live one again.
    qpos = live.copy()
    status = _tick(gate, qpos, 2.0, state=TeleopState.RESUMING, lowstate=(np.full(31, 1.5), 0.0), resume=7)
    assert status.trigger == TRIGGER_RESUME and not status.waiting and np.allclose(qpos[HINGES], 1.5) and np.array_equal(qpos[:7], live[:7])
    # A trigger without a fresh LowState also replaces a running ramp by the hold.
    qpos = live.copy()
    status = _tick(gate, qpos, 2.1, state=TeleopState.PAUSED, lowstate=None, resume=7)
    assert not status.waiting and status.converging
    after_ramp_tick = qpos.copy()
    qpos = live.copy()
    status = _tick(gate, qpos, 2.2, state=TeleopState.RESUMING, lowstate=None, resume=8)
    assert status.waiting and not gate.active and np.array_equal(qpos, after_ramp_tick) and gate.deferred == 2


def test_resume_count_is_echoed_only_after_the_ramp_of_that_request():
    gate = _gate()
    lowstate = (np.full(31, 1.0), 0.01)
    assert _tick(gate, _qpos(2.0), 0.0, state=TeleopState.PAUSED, lowstate=lowstate, resume=41).converged_resume_count == 0
    status = _tick(gate, _qpos(2.0), 1.0, state=TeleopState.RESUMING, lowstate=lowstate, resume=42)
    assert status.converging and status.converged_resume_count == 0
    assert _tick(gate, _qpos(2.0), 1.5, state=TeleopState.RESUMING, lowstate=lowstate, resume=42).converged_resume_count == 0
    status = _tick(gate, _qpos(2.0), 2.0, state=TeleopState.RESUMING, lowstate=lowstate, resume=42)
    assert not status.converging and status.converged_resume_count == 42
    # The echo stays while the headset goes Active and pauses; the next request is not acknowledged early.
    assert _tick(gate, _qpos(2.0), 2.1, state=TeleopState.ACTIVE, lowstate=lowstate, resume=42).converged_resume_count == 42
    assert _tick(gate, _qpos(2.0), 2.2, state=TeleopState.PAUSED, lowstate=lowstate, resume=42).converged_resume_count == 42
    status = _tick(gate, _qpos(3.0), 3.0, state=TeleopState.RESUMING, lowstate=lowstate, resume=43)
    assert status.trigger == TRIGGER_RESUME and status.converged_resume_count == 42
    # Paused before the ramp ended: a ramp that finishes outside RESUMING acknowledges nothing.
    assert _tick(gate, _qpos(3.0), 3.5, state=TeleopState.PAUSED, lowstate=lowstate, resume=43).converged_resume_count == 42
    status = _tick(gate, _qpos(3.0), 4.0, state=TeleopState.PAUSED, lowstate=lowstate, resume=43)
    assert not status.converging and status.converged_resume_count == 42
    # A new id inside RESUMING (the PAUSED records were never sampled) is a new request and restarts the ramp.
    gate = _gate()
    assert _tick(gate, _qpos(2.0), 0.0, state=TeleopState.RESUMING, lowstate=lowstate, resume=9).trigger == TRIGGER_RESUME
    status = _tick(gate, _qpos(2.0), 0.5, state=TeleopState.RESUMING, lowstate=lowstate, resume=10)
    assert status.trigger == TRIGGER_RESUME and gate.started == 2 and status.converged_resume_count == 0
    assert _tick(gate, _qpos(2.0), 1.5, state=TeleopState.RESUMING, lowstate=lowstate, resume=10).converged_resume_count == 10


def test_resume_echo_edge_cases():
    lowstate = (np.full(31, 1.0), 0.0)
    # Operator already at the robot: the ramp ends on its first tick and the request is acknowledged with it.
    gate = _gate(max_rate=0.5)
    status = _tick(gate, _qpos(1.01), 0.0, state=TeleopState.RESUMING, lowstate=lowstate, resume=3)
    assert status.trigger == TRIGGER_RESUME and not status.converging and status.converged_resume_count == 3
    # Skeleton lost: the echo restarts at 0 (a restarted headset must not match an earlier session's id).
    gate.reset()
    assert _tick(gate, _qpos(1.01), 1.0, state=TeleopState.ACTIVE, edge=True, lowstate=lowstate, resume=3).converged_resume_count == 0
    # A headset without the field (0) is never acknowledged; its RESUMING edge still triggers.
    gate = _gate()
    status = _tick(gate, _qpos(1.0), 0.0, state=TeleopState.RESUMING, lowstate=lowstate, resume=0)
    assert status.trigger == TRIGGER_RESUME and status.converged_resume_count == 0
    # A low-level entry ramp while the headset is still Resuming delays the acknowledgement until it ends.
    gate = _gate()
    _tick(gate, _qpos(2.0), 0.0, state=TeleopState.PAUSED, mode=WALK, lowstate=lowstate, resume=1)
    assert _tick(gate, _qpos(2.0), 1.0, state=TeleopState.RESUMING, mode=WALK, lowstate=lowstate, resume=2).converging
    status = _tick(gate, _qpos(2.0), 1.9, state=TeleopState.RESUMING, mode=LOW, lowstate=lowstate, resume=2)
    assert status.trigger == TRIGGER_LOW_LEVEL and status.converged_resume_count == 0
    assert _tick(gate, _qpos(2.0), 2.5, state=TeleopState.RESUMING, mode=LOW, lowstate=lowstate, resume=2).converged_resume_count == 0
    assert _tick(gate, _qpos(2.0), 2.9, state=TeleopState.RESUMING, mode=LOW, lowstate=lowstate, resume=2).converged_resume_count == 2


def test_disabled_gate_never_shapes_but_still_tracks_state():
    gate = _gate(enabled=False)
    qpos = _qpos(2.0)
    status = _tick(gate, qpos, 0.0, state=TeleopState.RESUMING, lowstate=(np.zeros(31), 0.0))
    assert not status.converging and status.trigger is None and np.allclose(qpos[HINGES], 2.0)
    # No ramp to wait for: a request is acknowledged at once, also without any LowState.
    status = _tick(gate, qpos, 0.1, state=TeleopState.RESUMING, lowstate=None, resume=6)
    assert not status.converging and status.publish and not status.waiting and status.converged_resume_count == 6
    assert np.allclose(qpos[HINGES], 2.0)


def test_hand_command_is_rate_limited_only_while_converging():
    gate = _gate(hand_rate=1.0)
    assert np.allclose(gate.shape_hand(np.full(12, 0.0), 0.0), 0.0)
    assert np.allclose(gate.shape_hand(np.full(12, 1.0), 0.1), 1.0)  # idle: pass-through
    assert gate.shape_hand(None, 0.2) is None
    _tick(gate, _qpos(1.0), 0.2, state=TeleopState.RESUMING, lowstate=(np.zeros(31), 0.0))
    limited = gate.shape_hand(np.full(12, 0.0), 0.3)
    assert np.allclose(limited, 0.9)  # 1.0/s cap from the last pass-through value


def test_hand_command_is_held_while_a_trigger_waits_for_lowstate():
    gate = _gate(hand_rate=1.0)
    # First start with no LowState: there is no earlier command, so nothing is published.
    _tick(gate, _qpos(1.0), 0.0, state=TeleopState.RESUMING, lowstate=None)
    assert gate.shape_hand(np.full(12, 1.0), 0.0) is None
    gate = _gate(hand_rate=1.0)
    assert np.allclose(gate.shape_hand(np.full(12, 0.2), 0.0), 0.2)  # idle: pass-through
    _tick(gate, _qpos(1.0), 0.1, state=TeleopState.RESUMING, lowstate=None)
    assert np.allclose(gate.shape_hand(np.full(12, 1.0), 0.1), 0.2)  # waiting: the last command is held
    assert np.allclose(gate.shape_hand(np.full(12, 1.0), 5.0), 0.2)
    # The ramp starts: the command leaves the held value at the rate cap, not with the 5 s of waiting.
    _tick(gate, _qpos(1.0), 5.05, state=TeleopState.RESUMING, lowstate=(np.zeros(31), 0.0))
    assert np.allclose(gate.shape_hand(np.full(12, 1.0), 5.05), 0.25)


def test_low_level_modes_follow_the_binding_constants():
    class Sdk:
        ROBOT_STATE_LOW = 30
        ROBOT_STATE_WALK_LOW = 60

    assert low_level_modes_from_sdk(Sdk) == frozenset({30, 60})
    assert low_level_modes_from_sdk(object()) == frozenset({3, 6})
    with pytest.raises(ValueError):
        _tick(_gate(), _qpos(1.0), 0.0, state=TeleopState.RESUMING, lowstate=(np.zeros(30), 0.0))
