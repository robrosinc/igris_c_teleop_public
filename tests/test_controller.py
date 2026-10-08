"""Controller button edges and the event -> control-mode table."""

from __future__ import annotations

from vr_link.types import VRInputState

from igris_c_gmr.controller import EVENT_TO_CONTROL_MODE, ConnectionSessions, ControllerEvents
from vr_fixtures import click


def test_button_mapping_edges():
    """Right B -> lowcmd_start, R stick -> stop, L stick -> go_home; connect alone fires nothing."""

    events = ControllerEvents()
    disconnected = VRInputState(connected=False)

    assert events.update(False, disconnected) == {}
    assert events.update(True, click()) == {}
    assert events.update(True, click(b=True)) == {"lowcmd_start": True}
    assert events.update(True, click(b=True)) == {}  # held: no repeat
    assert events.update(True, click()) == {}
    assert events.update(True, click(b=True)) == {"lowcmd_start": True}  # pressed again
    assert events.update(True, click(right=True)) == {"stop": True}
    assert events.update(True, click(right=True)) == {}
    assert events.update(True, click()) == {}
    assert events.update(True, click(left=True)) == {"go_home": True}
    assert events.update(False, disconnected) == {}  # disconnect fires nothing
    assert events.update(True, click()) == {}  # reconnect fires nothing


def test_button_held_across_disconnect_does_not_refire():
    events = ControllerEvents()
    assert events.update(True, click(b=True)) == {"lowcmd_start": True}
    assert events.update(False, VRInputState(connected=False)) == {}
    assert events.update(True, click(b=True)) == {"lowcmd_start": True}  # released while disconnected -> new edge


def test_no_auto_start_on_connect():
    events = ControllerEvents()
    assert not hasattr(events, "auto_start")
    assert events.update(True, click()) == {}
    assert events.update(False, VRInputState(connected=False)) == {}
    assert events.update(True, click()) == {}


def test_simultaneous_and_disabled():
    events = ControllerEvents()
    assert events.update(True, click(right=True, left=True, b=True)) == {"lowcmd_start": True, "stop": True, "go_home": True}

    disabled = ControllerEvents(enabled=False)
    assert disabled.update(True, click(right=True)) == {}
    disabled.enabled = True
    assert disabled.update(True, click(right=True)) == {}  # edge already consumed while disabled


def test_connection_sessions_count_reconnections_only():
    sessions = ConnectionSessions()
    assert sessions.session_id == 0
    assert sessions.update(False) == 0  # never connected yet
    assert sessions.update(True) == 1  # first connection
    assert sessions.update(True) == 1  # staying connected is not a new session
    assert sessions.update(False) == 1  # losing the skeleton keeps the id until it returns
    assert sessions.update(False) == 1
    assert sessions.update(True) == 2  # reconnected: recorders re-seed the toggle parity
    assert sessions.update(True) == 2


def test_control_mode_table_matches_gmt_bridge():
    assert EVENT_TO_CONTROL_MODE == {
        "lowcmd_start": ("CONTROL_MODE_CMD_LOW_LEVEL_JOINT_CONTROL", ""),
        "stop": ("CONTROL_MODE_CMD_JOINT_POSITION_HOLD", ""),
        "go_home": ("CONTROL_MODE_CMD_MOTION_PRESET", "HOME"),
    }
    assert ControllerEvents.EVENT_NAMES == tuple(EVENT_TO_CONTROL_MODE)
