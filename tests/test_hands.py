"""Normalized 12-motor finger command: grip, hand tracking, auto arbitration, hold."""

from __future__ import annotations

import numpy as np
import pytest
from general_motion_retargeting.hand_retarget import igris_c_hand_command as hand_cmd
from vr_link.types import ControllerInput

from igris_c_teleop.hands import HandTargets
from vr_fixtures import curl_from_tip_z, make_hand, make_state

RIGHT = hand_cmd.RIGHT_MOTOR_SLOTS
LEFT = hand_cmd.LEFT_MOTOR_SLOTS
FULL = hand_cmd.GRIP_FULL_COMMAND  # 0.6 at a fully squeezed grip


def test_grip_hand_targets_fill_the_flexion_motors():
    hands = HandTargets("grip")
    assert hands.enabled and hands.tracked == (False, False) and hands.motor_ids == hand_cmd.MOTOR_IDS
    command = hands.update(make_state(left=ControllerInput(grip=0.25), right=ControllerInput(grip=1.0)))
    assert command.shape == (12,)
    np.testing.assert_allclose(command, hand_cmd.grip_to_normalized(0.25, 1.0))
    np.testing.assert_allclose(command[RIGHT], FULL)
    np.testing.assert_allclose(command[LEFT], 0.25 * FULL)
    assert hands.active_sources == ("grip", "grip")
    clamped = hands.update(make_state(left=ControllerInput(grip=-1.0), right=ControllerInput(grip=2.0)))
    np.testing.assert_allclose(clamped, hand_cmd.grip_to_normalized(0.0, 1.0))
    forced = hands.update(make_state(left_hand=make_hand(z=0.9), right_hand=make_hand(z=0.9), right=ControllerInput(grip=0.5)))
    np.testing.assert_allclose(forced[RIGHT], 0.5 * FULL)
    assert hands.tracked == (False, False)

    none = HandTargets("none")
    assert not none.enabled and none.update(make_state()) is None and none.active_sources == ("hold", "hold")
    with pytest.raises(ValueError):
        HandTargets("fist")


def test_auto_source_arbitrates_per_hand():
    hands = HandTargets("auto", curl_fn=curl_from_tip_z)
    assert hands.enabled and hands.tracked == (False, False) and hands.active_sources == ("hold", "hold")

    both = hands.update(make_state(left_hand=make_hand(z=0.1), right_hand=make_hand(z=0.2), left=ControllerInput(grip=1.0)))
    np.testing.assert_allclose(both[LEFT], 0.1)
    np.testing.assert_allclose(both[RIGHT], 0.2)
    assert hands.tracked == (True, True) and hands.active_sources == ("hand", "hand")

    held = hands.update(make_state(left_hand=make_hand(z=0.3), right_hand=make_hand(tracked=False, z=9.0)))
    np.testing.assert_allclose(held[LEFT], 0.3)
    np.testing.assert_allclose(held[RIGHT], 0.2)
    assert hands.tracked == (True, False) and hands.active_sources == ("hand", "hold")

    picked = hands.update(make_state(left_hand=make_hand(z=0.3), right_hand=make_hand(tracked=False), right=ControllerInput(grip=0.5)))
    np.testing.assert_allclose(picked[RIGHT], 0.5 * FULL)
    assert hands.active_sources == ("hand", "grip")

    absent = hands.update(make_state(left=ControllerInput(grip=0.1)))
    assert hands.active_sources == ("grip", "grip")
    np.testing.assert_allclose(absent[LEFT], 0.1 * FULL)
    np.testing.assert_allclose(absent[RIGHT], 0.0)

    short = hands.update(make_state(left_hand=make_hand(z=0.9, count=21), right_hand=make_hand(z=0.4)))
    np.testing.assert_allclose(short[LEFT], 0.1 * FULL)
    np.testing.assert_allclose(short[RIGHT], 0.4)
    assert hands.active_sources == ("hold", "hand") and hands.tracked == (False, True)

    hands.reset()
    assert hands.tracked == (False, False) and hands.active_sources == ("hold", "hold")
    after = hands.update(make_state(left_hand=make_hand(tracked=False), right_hand=make_hand(tracked=False)))
    np.testing.assert_allclose(after, short)
    assert hands.active_sources == ("hold", "hold")


def test_forced_hand_source_holds_while_untracked():
    hands = HandTargets("hand", curl_fn=curl_from_tip_z)
    command = hands.update(make_state(left_hand=make_hand(tracked=False), right_hand=None, right=ControllerInput(grip=1.0)))
    np.testing.assert_allclose(command, 0.0)
    assert hands.active_sources == ("hold", "hold") and hands.tracked == (False, False)
    command = hands.update(make_state(left_hand=make_hand(z=0.5), right_hand=make_hand(z=0.6)))
    np.testing.assert_allclose(command[LEFT], 0.5)
    np.testing.assert_allclose(command[RIGHT], 0.6)
    assert hands.active_sources == ("hand", "hand")


def test_hand_source_holds_when_curl_fails():
    hands = HandTargets("auto", curl_fn=lambda keypoints, calibration: None)
    command = hands.update(make_state(left_hand=make_hand(), right_hand=make_hand()))
    np.testing.assert_allclose(command, 0.0)
    assert hands.active_sources == ("hold", "hold") and hands.tracked == (False, False)
