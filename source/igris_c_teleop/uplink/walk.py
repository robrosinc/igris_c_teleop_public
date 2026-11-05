"""Thumbsticks -> walk velocity (``cmd_vel``) for the robot's Low Level Walkmode.

Same layout and numbers as the robot's gamepad node (igris_c ``igris_c_joystick/src/joystick_publisher_node.cpp``,
2026-11-04): left stick forward/back walks forward/backward (``vx``), left stick left/right steps
left/right (``vy``), right stick left/right turns left/right (``yaw_rate``, counter-clockwise
positive). Maxima 0.5 m/s and 0.8 rad/s at full deflection, deadzone 0.1: radial on the left
stick (the planar command keeps its direction and starts at 0 just outside the deadzone), scalar on
the right stick X. Nothing is commanded while the headset is not Active. The robot clamps each
axis to +-1 (robotHandler.cpp) and the walk policy applies its own deadband and filter on top.

Axis convention assumed from the VRS1 contract (``ControllerInput.thumbstick`` = (x, y), each in
[-1, 1]) and the OpenXR thumbstick frame (+x right, +y forward): left ``y`` -> ``vx``, left ``x``
-> ``vy`` with +x (right) giving a negative (rightward) ``vy``, right ``x`` -> ``yaw_rate`` with +x
giving a negative (clockwise) rate. The gamepad node negates evdev's up/left-negative axes for the
same result. Verify on the headset before trusting the signs (``--walk_invert_lateral`` /
``--walk_invert_yaw`` flip them; ``--measure_fps`` logs the command).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from vr_link.types import VRInputState

DEFAULT_MAX_LINEAR = 0.5    # m/s at full left-stick deflection (gamepad kMaxLinear)
DEFAULT_MAX_YAW_RATE = 0.8  # rad/s at full right-stick deflection (gamepad kMaxAngular)
DEFAULT_DEADZONE = 0.1      # normalized deadzone per stick (gamepad kDeadzone)


@dataclass(frozen=True)
class WalkVelocity:
    """Planar base velocity command in the robot's body frame (igris_c_sdk TwistStamped on ``cmd_vel``)."""

    vx: float = 0.0        # m/s, forward positive
    vy: float = 0.0        # m/s, left positive
    yaw_rate: float = 0.0  # rad/s, counter-clockwise positive

    @property
    def is_zero(self) -> bool:
        return self.vx == 0.0 and self.vy == 0.0 and self.yaw_rate == 0.0


WALK_STOP = WalkVelocity()


def _clamp_unit(value: float) -> float:
    v = float(value)
    return 0.0 if not math.isfinite(v) else max(-1.0, min(1.0, v))


def radial_deadzone(x: float, y: float, deadzone: float) -> tuple[float, float]:
    """Gamepad ``ApplyDeadzone``: zero inside the circle, outside rescaled so the magnitude starts at 0."""

    x, y = _clamp_unit(x), _clamp_unit(y)
    magnitude = math.hypot(x, y)
    if magnitude <= deadzone:
        return 0.0, 0.0
    scale = (magnitude - deadzone) / (1.0 - deadzone) / magnitude
    return x * scale, y * scale


def scalar_deadzone(value: float, deadzone: float) -> float:
    """Gamepad ``Deadzone1D``: zero inside, outside rescaled to [-1, 1] continuously."""

    v = _clamp_unit(value)
    if abs(v) <= deadzone:
        return 0.0
    return math.copysign((abs(v) - deadzone) / (1.0 - deadzone), v)


class ThumbstickWalk:
    """VRS1 thumbsticks -> ``WalkVelocity``; ``WALK_STOP`` whenever the headset is not Active."""

    def __init__(
        self,
        *,
        max_linear: float = DEFAULT_MAX_LINEAR,
        max_yaw_rate: float = DEFAULT_MAX_YAW_RATE,
        deadzone: float = DEFAULT_DEADZONE,
        invert_lateral: bool = False,
        invert_yaw: bool = False,
    ) -> None:
        if not (max_linear >= 0.0 and max_yaw_rate >= 0.0):
            raise ValueError("walk maxima must be non-negative")
        if not 0.0 <= deadzone < 1.0:
            raise ValueError("walk deadzone must lie in [0, 1)")
        self.max_linear = float(max_linear)
        self.max_yaw_rate = float(max_yaw_rate)
        self.deadzone = float(deadzone)
        self.lateral_sign = 1.0 if invert_lateral else -1.0  # left stick +x (right) -> step right -> negative vy
        self.yaw_sign = 1.0 if invert_yaw else -1.0           # right stick +x (right) -> clockwise -> negative yaw rate

    def command(self, state: VRInputState, active: bool) -> WalkVelocity:
        if not active:
            return WALK_STOP
        left_x, left_y = radial_deadzone(*state.left_input.thumbstick, self.deadzone)
        right_x = scalar_deadzone(state.right_input.thumbstick[0], self.deadzone)
        return WalkVelocity(
            vx=self.max_linear * left_y,
            vy=self.lateral_sign * self.max_linear * left_x,
            yaw_rate=self.yaw_sign * self.max_yaw_rate * right_x,
        )


__all__ = [
    "DEFAULT_DEADZONE",
    "DEFAULT_MAX_LINEAR",
    "DEFAULT_MAX_YAW_RATE",
    "WALK_STOP",
    "ThumbstickWalk",
    "WalkVelocity",
    "radial_deadzone",
    "scalar_deadzone",
]
