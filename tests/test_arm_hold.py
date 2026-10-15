"""ArmHold: engage on hand loss / invalid chain, hold across torso motion, debounce and ramp the release."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from vr_link.types import BODY_JOINT_NAMES, TrackedPose

from igris_c_teleop.shaping.arm_hold import ArmHold, arm_chain_valid, arm_qpos_addresses
from vr_fixtures import make_hand, make_state

LEFT_ARM_HINGES = list(range(7 + 15, 7 + 22))
RIGHT_ARM_HINGES = list(range(7 + 22, 7 + 29))


def test_arm_addresses_match_the_serial_joint_order(model):
    assert arm_qpos_addresses(model, "left").tolist() == LEFT_ARM_HINGES
    assert arm_qpos_addresses(model, "right").tolist() == RIGHT_ARM_HINGES


def _invalidate(state, name):
    joints = list(state.body_joints)
    index = BODY_JOINT_NAMES.index(name)
    joints[index] = TrackedPose(valid=False)
    return replace(state, body_joints=tuple(joints))


def test_hold_requested_from_arbiter_or_invalid_chain():
    state = make_state(left_hand=make_hand(tracked=False), right_hand=make_hand(tracked=True))
    assert arm_chain_valid(state, "left") and arm_chain_valid(state, "right")
    assert ArmHold.hold_requested(state, "left", "hold") == (True, "hand tracking lost")
    assert ArmHold.hold_requested(state, "right", "hand") == (False, "")
    assert ArmHold.hold_requested(state, "right", "grip") == (False, "")
    broken = _invalidate(state, "Right_Wrist")
    assert not arm_chain_valid(broken, "right") and arm_chain_valid(broken, "left")
    assert ArmHold.hold_requested(broken, "right", "hand") == (True, "arm body joints invalid")


def test_hold_freezes_the_arm_debounces_the_release_and_ramps_back(model):
    hold = ArmHold(model, release_stable_s=0.2, release_rate=10.0, release_duration_s=0.5)
    qpos = np.zeros(38)
    qpos[LEFT_ARM_HINGES] = 1.0
    qpos[RIGHT_ARM_HINGES] = 2.0
    live_right = 2.0
    # Tick 0: left hand lost -> left arm held at 1.0; right stays live.
    assert hold.update(qpos, now=0.0, requests={"left": (True, "hand tracking lost"), "right": (False, "")}) == (True, False)
    assert hold.engaged == 1
    # The live reference moves; the held arm does not, the free arm does.
    qpos[LEFT_ARM_HINGES] = 5.0
    qpos[RIGHT_ARM_HINGES] = live_right = 3.0
    hold.update(qpos, now=0.1, requests={"left": (True, "hand tracking lost"), "right": (False, "")})
    assert np.allclose(qpos[LEFT_ARM_HINGES], 1.0) and np.allclose(qpos[RIGHT_ARM_HINGES], live_right)
    # Tracking returns at t=0.2: still held until it is stable for 0.2 s.
    qpos[LEFT_ARM_HINGES] = 5.0
    assert hold.update(qpos, now=0.2, requests={"left": (False, ""), "right": (False, "")}) == (True, False)
    assert np.allclose(qpos[LEFT_ARM_HINGES], 1.0)
    qpos[LEFT_ARM_HINGES] = 5.0
    assert hold.update(qpos, now=0.3, requests={"left": (False, ""), "right": (False, "")}) == (True, False)
    # A flicker back to lost restarts the debounce without re-engaging.
    qpos[LEFT_ARM_HINGES] = 5.0
    hold.update(qpos, now=0.35, requests={"left": (True, "hand tracking lost"), "right": (False, "")})
    assert hold.engaged == 1 and np.allclose(qpos[LEFT_ARM_HINGES], 1.0)
    for t in (0.4, 0.5):
        qpos[LEFT_ARM_HINGES] = 5.0
        hold.update(qpos, now=t, requests={"left": (False, ""), "right": (False, "")})
    assert hold.holding == (True, False)
    # Stable for 0.2 s (tracked since t=0.4): released, ramp starts at the held pose (not at the live 5.0).
    qpos[LEFT_ARM_HINGES] = 5.0
    assert hold.update(qpos, now=0.65, requests={"left": (False, ""), "right": (False, "")}) == (False, False)
    assert hold.released == 1 and hold.releasing == (True, False) and np.allclose(qpos[LEFT_ARM_HINGES], 1.0)
    qpos[LEFT_ARM_HINGES] = 5.0
    hold.update(qpos, now=0.75, requests={"left": (False, ""), "right": (False, "")})
    assert 1.0 < qpos[LEFT_ARM_HINGES][0] < 5.0  # mid-ramp
    for step in range(1, 20):  # 0.8 .. 1.75 s at 50 ms: the 0.5 s blend completes, the 10 rad/s cap never binds
        qpos[LEFT_ARM_HINGES] = 5.0
        hold.update(qpos, now=0.75 + 0.05 * step, requests={"left": (False, ""), "right": (False, "")})
    assert hold.releasing == (False, False) and np.allclose(qpos[LEFT_ARM_HINGES], 5.0)


def test_hold_during_release_ramp_freezes_the_ramp_output(model):
    hold = ArmHold(model, release_stable_s=0.0, release_rate=1.0, release_duration_s=0.0)
    qpos = np.zeros(38)
    hold.update(qpos, now=0.0, requests={"left": (True, "x"), "right": (False, "")})
    qpos[LEFT_ARM_HINGES] = 4.0
    hold.update(qpos, now=0.1, requests={"left": (False, ""), "right": (False, "")})  # released, ramp from 0 at 1 rad/s
    qpos[LEFT_ARM_HINGES] = 4.0
    hold.update(qpos, now=0.2, requests={"left": (False, ""), "right": (False, "")})
    mid = qpos[LEFT_ARM_HINGES].copy()
    assert 0.0 < mid[0] < 4.0
    qpos[LEFT_ARM_HINGES] = 4.0
    hold.update(qpos, now=0.3, requests={"left": (True, "x"), "right": (False, "")})
    assert np.allclose(qpos[LEFT_ARM_HINGES], mid) and hold.holding == (True, False)


def test_disabled_and_reset(model):
    hold = ArmHold(model, enabled=False)
    qpos = np.ones(38)
    assert hold.update(qpos, now=0.0, requests={"left": (True, "x"), "right": (True, "x")}) == (False, False)
    assert np.allclose(qpos, 1.0)
    hold = ArmHold(model)
    hold.update(qpos, now=0.0, requests={"left": (True, "x"), "right": (False, "")})
    hold.reset()
    assert hold.holding == (False, False) and hold.releasing == (False, False)
