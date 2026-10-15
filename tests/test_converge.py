"""ConvergeGate: triggers (teleop_state, legacy connect, low-level entry), ramping, skip without LowState."""

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


def _tick(gate, qpos, now, *, state=TeleopState.ACTIVE, edge=False, mode=None, lowstate=None):
    return gate.update(qpos, now=now, teleop_state=state, connected_edge=edge, robot_mode=mode, lowstate=lowstate)


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
    assert gate.started == 1 and gate.skipped == 0
    # Staying in RESUMING does not re-trigger; a new RESUMING after ACTIVE does.
    assert _tick(gate, _qpos(2.0), 3.0, state=TeleopState.RESUMING, lowstate=lowstate).trigger == TRIGGER_RESUME
    assert _tick(gate, _qpos(2.0), 3.1, state=TeleopState.RESUMING, lowstate=lowstate).trigger is None
    assert gate.started == 2


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


def test_trigger_without_fresh_lowstate_is_skipped_and_passes_through():
    gate = _gate()
    qpos = _qpos(2.0)
    status = _tick(gate, qpos, 0.0, state=TeleopState.RESUMING, lowstate=None)
    assert not status.converging and status.skipped == TRIGGER_RESUME and np.allclose(qpos[HINGES], 2.0)
    status = _tick(gate, _qpos(2.0), 1.0, state=TeleopState.ACTIVE)
    status = _tick(gate, _qpos(2.0), 2.0, state=TeleopState.RESUMING, lowstate=(np.zeros(31), 0.9))  # too old
    assert status.skipped == TRIGGER_RESUME and gate.skipped == 2 and gate.started == 0


def test_disabled_gate_never_shapes_but_still_tracks_state():
    gate = _gate(enabled=False)
    qpos = _qpos(2.0)
    status = _tick(gate, qpos, 0.0, state=TeleopState.RESUMING, lowstate=(np.zeros(31), 0.0))
    assert not status.converging and status.trigger is None and np.allclose(qpos[HINGES], 2.0)


def test_hand_command_is_rate_limited_only_while_converging():
    gate = _gate(hand_rate=1.0)
    assert np.allclose(gate.shape_hand(np.full(12, 0.0), 0.0), 0.0)
    assert np.allclose(gate.shape_hand(np.full(12, 1.0), 0.1), 1.0)  # idle: pass-through
    assert gate.shape_hand(None, 0.2) is None
    _tick(gate, _qpos(1.0), 0.2, state=TeleopState.RESUMING, lowstate=(np.zeros(31), 0.0))
    limited = gate.shape_hand(np.full(12, 0.0), 0.3)
    assert np.allclose(limited, 0.9)  # 1.0/s cap from the last pass-through value


def test_low_level_modes_follow_the_binding_constants():
    class Sdk:
        ROBOT_STATE_LOW = 30
        ROBOT_STATE_WALK_LOW = 60

    assert low_level_modes_from_sdk(Sdk) == frozenset({30, 60})
    assert low_level_modes_from_sdk(object()) == frozenset({3, 6})
    with pytest.raises(ValueError):
        _tick(_gate(), _qpos(1.0), 0.0, state=TeleopState.RESUMING, lowstate=(np.zeros(30), 0.0))
