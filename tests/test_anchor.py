"""OperatorFrameAnchor: a Start/Resume continues from the last retargeted heading and foot position."""

from __future__ import annotations

import math

import numpy as np
from vr_link.types import TeleopState

from igris_c_teleop.shaping.anchor import OperatorFrameAnchor


def _skeleton(yaw_deg: float = 0.0, origin=(0.0, 0.0), lean_quat=(1.0, 0.0, 0.0, 0.0)):
    """A standing stick figure facing ``yaw_deg`` at ``origin`` (x forward, y left, z up)."""

    local = {
        "Pelvis": (0.0, 0.0, 0.9),
        "Left_Hip": (0.0, 0.1, 0.85),
        "Right_Hip": (0.0, -0.1, 0.85),
        "Left_Foot": (0.05, 0.12, 0.05),
        "Right_Foot": (0.05, -0.12, 0.05),
        "Head": (0.02, 0.0, 1.6),
    }
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    half = math.radians(yaw_deg) / 2.0
    cw, sw = math.cos(half), math.sin(half)
    w, x, y, z = lean_quat
    quat = np.array([cw * w - sw * z, cw * x - sw * y, cw * y + sw * x, cw * z + sw * w])
    return {
        name: [np.array([c * p[0] - s * p[1] + origin[0], s * p[0] + c * p[1] + origin[1], p[2]]), quat.copy()]
        for name, p in local.items()
    }


def _same(a, b, tol=1e-9):
    return all(np.allclose(a[k][0], b[k][0], atol=tol) and np.allclose(np.abs(a[k][1] @ b[k][1]), 1.0, atol=tol) for k in a)


def _update(anchor, human, state, count=0):
    return anchor.update(human, teleop_state=state, resume_count=count)


def test_identity_until_the_second_request():
    anchor = OperatorFrameAnchor()
    first = _skeleton(yaw_deg=20.0, origin=(1.0, 2.0))
    # First-ever Start: nothing to continue from, the skeleton passes unchanged (as a new dict).
    out = _update(anchor, first, TeleopState.RESUMING, 5)
    assert _same(out, first) and out is not first and anchor.realigned == 0 and anchor.yaw == 0.0
    assert _same(_update(anchor, first, TeleopState.ACTIVE, 5), first)


def test_turn_and_walk_while_paused_do_not_reach_the_retargeting():
    anchor = OperatorFrameAnchor()
    before = _skeleton(yaw_deg=10.0, origin=(0.5, -0.3))
    _update(anchor, before, TeleopState.ACTIVE, 1)
    frozen = _update(anchor, before, TeleopState.PAUSED, 1)  # frozen snapshot re-sent while paused
    # The operator turned 90 deg and walked away, then resumed in the same posture.
    after = _skeleton(yaw_deg=100.0, origin=(2.0, 1.5))
    out = _update(anchor, after, TeleopState.RESUMING, 2)
    assert anchor.realigned == 1 and _same(out, frozen, tol=1e-9)
    assert math.isclose(math.degrees(anchor.yaw), -90.0, abs_tol=1e-9)
    # The transform is fixed from here on: later motion shows up in full, in the anchored frame.
    moved = _skeleton(yaw_deg=130.0, origin=(2.0, 1.5))
    out = _update(anchor, moved, TeleopState.ACTIVE, 2)
    assert _same(out, _skeleton(yaw_deg=40.0, origin=(0.5, -0.3)), tol=1e-9) and anchor.realigned == 1
    # Staying in RESUMING / ACTIVE never re-anchors; z and the input dict are never touched.
    assert out["Head"][0][2] == moved["Head"][0][2] and np.allclose(after["Pelvis"][0][:2], _skeleton(100.0, (2.0, 1.5))["Pelvis"][0][:2])


def test_only_heading_and_position_are_aligned_not_the_posture():
    anchor = OperatorFrameAnchor()
    upright = _skeleton(yaw_deg=0.0)
    _update(anchor, upright, TeleopState.PAUSED, 1)
    # Resumes turned 45 deg AND leaning forward (30 deg pitch about the body's y axis) with a lower head.
    lean = (math.cos(math.radians(15.0)), 0.0, math.sin(math.radians(15.0)), 0.0)
    leaning = _skeleton(yaw_deg=45.0, origin=(1.0, 0.0), lean_quat=lean)
    leaning["Head"][0][2] = 1.2
    out = _update(anchor, leaning, TeleopState.RESUMING, 2)
    # Heading and feet are where they were; the lean and the height pass through.
    expected = _skeleton(yaw_deg=0.0, lean_quat=lean)
    expected["Head"][0][2] = 1.2
    assert _same(out, expected, tol=1e-9)


def test_new_id_inside_resuming_and_reconnect_are_requests_too():
    anchor = OperatorFrameAnchor()
    _update(anchor, _skeleton(0.0), TeleopState.RESUMING, 1)
    # The PAUSED records in between were never sampled: a new id while RESUMING re-anchors.
    out = _update(anchor, _skeleton(30.0, (0.4, 0.0)), TeleopState.RESUMING, 2)
    assert anchor.realigned == 1 and _same(out, _skeleton(0.0), tol=1e-9)
    # Skeleton lost and back in RESUMING with the same id (e.g. another tracking origin after a headset restart).
    anchor.reset()
    out = _update(anchor, _skeleton(-120.0, (5.0, 5.0)), TeleopState.RESUMING, 2)
    assert anchor.realigned == 2 and _same(out, _skeleton(0.0), tol=1e-9)
    # A reconnect straight into ACTIVE is not a request: the frame stays as it is.
    anchor.reset()
    out = _update(anchor, _skeleton(-100.0, (5.0, 5.0)), TeleopState.ACTIVE, 2)
    assert anchor.realigned == 2 and _same(out, _skeleton(20.0), tol=1e-9)


def test_disabled_and_degenerate_skeleton():
    anchor = OperatorFrameAnchor(enabled=False)
    first, second = _skeleton(0.0), _skeleton(90.0, (1.0, 1.0))
    _update(anchor, first, TeleopState.PAUSED, 1)
    assert _update(anchor, second, TeleopState.RESUMING, 2) is second and anchor.realigned == 0
    # Hip line too short to give a heading: the request is not re-anchored and the old transform stays.
    anchor = OperatorFrameAnchor()
    _update(anchor, first, TeleopState.PAUSED, 1)
    flat = _skeleton(90.0, (1.0, 1.0))
    flat["Right_Hip"][0][:] = flat["Left_Hip"][0]
    out = _update(anchor, flat, TeleopState.RESUMING, 2)
    assert anchor.realigned == 0 and anchor.yaw == 0.0 and np.allclose(out["Pelvis"][0], flat["Pelvis"][0])
