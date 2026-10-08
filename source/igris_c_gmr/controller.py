"""Controller button edge detection and the button -> robot control-mode mapping.

Same mapping as the retired ``igris_c_gmt_public`` ``gmr_control_mode_dds_bridge``:
right B -> low-level joint control, right thumbstick click -> joint position
hold, left thumbstick click -> motion preset ``HOME``.
"""

from __future__ import annotations

from vr_link.types import VRInputState

BUTTON_PRESSED_THRESHOLD = 0.5  # VRS1 buttons are floats (0/1 on current senders)

# event name -> (igris_c_sdk ControlModeCommandType constant name, preset_id)
EVENT_TO_CONTROL_MODE: dict[str, tuple[str, str]] = {
    "lowcmd_start": ("CONTROL_MODE_CMD_LOW_LEVEL_JOINT_CONTROL", ""),
    "stop": ("CONTROL_MODE_CMD_JOINT_POSITION_HOLD", ""),
    "go_home": ("CONTROL_MODE_CMD_MOTION_PRESET", "HOME"),
}


def _pressed(value: float) -> bool:
    return float(value) > BUTTON_PRESSED_THRESHOLD


class ControllerEvents:
    """Rising-edge detector turning VRS1 controller state into one-shot events.

    Right B -> ``lowcmd_start``, right thumbstick click -> ``stop``, left
    thumbstick click -> ``go_home``, each once per press while the skeleton is
    connected. Nothing fires on connect or disconnect: a lost link must not
    change the robot's control mode without operator intent. With
    ``enabled=False`` edges are still tracked but never reported.
    """

    EVENT_NAMES = tuple(EVENT_TO_CONTROL_MODE)

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._prev_active = {name: False for name in self.EVENT_NAMES}

    def update(self, connected: bool, state: VRInputState) -> dict[str, bool]:
        active = {
            "lowcmd_start": connected and _pressed(state.right_input.button_second),
            "stop": connected and _pressed(state.right_input.button_thumbstick),
            "go_home": connected and _pressed(state.left_input.button_thumbstick),
        }
        triggered = {name: True for name, is_active in active.items() if is_active and not self._prev_active[name]}
        self._prev_active = active
        return triggered if self.enabled else {}


class ConnectionSessions:
    """Count headset skeleton (re)connections for the ``vr/controller`` ``session_id``.

    The relayed ``record_toggle_count`` restarts when the headset reconnects.
    Recorders re-seed their parity rule when ``session_id`` changes
    (``vr_link.messages.teleop_dds.RecordToggleTracker``), so each lost ->
    connected transition must produce a new id. Starts at 0 (never connected).
    """

    def __init__(self) -> None:
        self.session_id = 0
        self._connected = False

    def update(self, connected: bool) -> int:
        connected = bool(connected)
        if connected and not self._connected:
            self.session_id += 1
        self._connected = connected
        return self.session_id


__all__ = ["BUTTON_PRESSED_THRESHOLD", "EVENT_TO_CONTROL_MODE", "ConnectionSessions", "ControllerEvents"]
