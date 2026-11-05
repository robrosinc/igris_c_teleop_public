"""Thumbsticks -> cmd_vel mapping, mirroring the robot's gamepad node: left stick vx/vy, right stick yaw."""

from __future__ import annotations

import math

import pytest
from vr_link.types import ControllerInput

from igris_c_teleop.uplink.walk import (
    DEFAULT_DEADZONE,
    DEFAULT_MAX_LINEAR,
    DEFAULT_MAX_YAW_RATE,
    WALK_STOP,
    ThumbstickWalk,
    WalkVelocity,
    radial_deadzone,
    scalar_deadzone,
)
from vr_fixtures import make_state


def sticks(left=(0.0, 0.0), right=(0.0, 0.0)):
    return make_state(left=ControllerInput(thumbstick=left), right=ControllerInput(thumbstick=right))


def test_numbers_match_the_gamepad_node():
    """igris_c_joystick joystick_publisher_node.cpp: kMaxLinear 0.5, kMaxAngular 0.8, kDeadzone 0.1."""

    assert (DEFAULT_MAX_LINEAR, DEFAULT_MAX_YAW_RATE, DEFAULT_DEADZONE) == (0.5, 0.8, 0.1)


def test_deadzones_follow_the_gamepad_functions():
    assert scalar_deadzone(0.0, 0.1) == 0.0 and scalar_deadzone(0.1, 0.1) == 0.0 and scalar_deadzone(-0.05, 0.1) == 0.0
    assert scalar_deadzone(1.0, 0.1) == 1.0 and scalar_deadzone(-1.0, 0.1) == -1.0
    assert scalar_deadzone(0.55, 0.1) == pytest.approx(0.5)
    assert scalar_deadzone(2.0, 0.1) == 1.0 and scalar_deadzone(float("nan"), 0.1) == 0.0
    assert radial_deadzone(0.07, 0.07, 0.1) == (0.0, 0.0)                       # inside the circle
    x, y = radial_deadzone(0.0, 0.55, 0.1)
    assert (x, y) == pytest.approx((0.0, 0.5))
    x, y = radial_deadzone(0.6, 0.8, 0.1)                                      # magnitude 1: direction kept
    assert math.hypot(x, y) == pytest.approx(1.0) and x / y == pytest.approx(0.75)


def test_left_stick_walks_and_steps_right_stick_turns():
    walk = ThumbstickWalk(deadzone=0.0)
    assert walk.command(sticks(left=(0.0, 1.0)), True) == WalkVelocity(vx=0.5)           # push forward
    assert walk.command(sticks(left=(0.0, -0.5)), True) == WalkVelocity(vx=-0.25)        # pull back
    assert walk.command(sticks(left=(-1.0, 0.0)), True) == WalkVelocity(vy=0.5)          # push left: step left (+y)
    assert walk.command(sticks(left=(1.0, 0.0)), True) == WalkVelocity(vy=-0.5)          # push right: step right
    assert walk.command(sticks(right=(-1.0, 0.0)), True) == WalkVelocity(yaw_rate=0.8)   # push left: turn left (CCW)
    assert walk.command(sticks(right=(1.0, 0.0)), True) == WalkVelocity(yaw_rate=-0.8)   # push right: turn right
    assert walk.command(sticks(right=(0.0, 1.0)), True) == WALK_STOP                     # right stick y is unused
    combined = walk.command(sticks(left=(0.6, 0.8), right=(-0.5, 0.0)), True)
    assert (combined.vx, combined.vy, combined.yaw_rate) == pytest.approx((0.4, -0.3, 0.4))


def test_inversions_and_active_gate():
    walk = ThumbstickWalk(deadzone=0.0, invert_lateral=True, invert_yaw=True)
    assert walk.command(sticks(left=(1.0, 0.0)), True).vy > 0.0
    assert walk.command(sticks(right=(1.0, 0.0)), True).yaw_rate > 0.0
    assert walk.command(sticks(left=(1.0, 1.0), right=(1.0, 0.0)), False) == WALK_STOP   # paused: nothing commanded
    assert WALK_STOP.is_zero and not WalkVelocity(vx=0.1).is_zero


def test_parameter_validation():
    with pytest.raises(ValueError):
        ThumbstickWalk(max_linear=-0.1)
    with pytest.raises(ValueError):
        ThumbstickWalk(deadzone=1.0)
    with pytest.raises(ValueError):
        ThumbstickWalk(deadzone=-0.1)
