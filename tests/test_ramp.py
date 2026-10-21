"""JointRamp / RateLimiter: smoothstep blend, rate cap, settle, pass-through."""

from __future__ import annotations

import numpy as np

from igris_c_teleop.shaping.ramp import JointRamp, RateLimiter, smoothstep


def test_smoothstep_shape():
    assert smoothstep(-1.0) == 0.0 and smoothstep(0.0) == 0.0 and smoothstep(1.0) == 1.0 and smoothstep(2.0) == 1.0
    assert smoothstep(0.5) == 0.5 and 0.0 < smoothstep(0.1) < 0.1  # eases in


def test_ramp_starts_at_start_and_ends_on_the_live_reference():
    indices = np.array([2, 3])
    ramp = JointRamp(indices, [0.0, 1.0], duration_s=1.0, max_rate=0.0, now=10.0)
    qpos = np.array([9.0, 9.0, 1.0, 2.0])  # live reference 1.0 / 2.0
    assert ramp.apply(qpos, 10.0) == 0.0 and qpos[2] == 0.0 and qpos[3] == 1.0  # tick 0: the start values
    qpos[2], qpos[3] = 1.0, 2.0
    assert ramp.apply(qpos, 10.5) == 0.5 and np.allclose(qpos[[2, 3]], [0.5, 1.5]) and not ramp.done
    qpos[2], qpos[3] = 1.0, 2.0
    assert ramp.apply(qpos, 11.0) == 1.0 and np.allclose(qpos[[2, 3]], [1.0, 2.0]) and ramp.done
    assert qpos[0] == 9.0 and qpos[1] == 9.0  # untouched joints


def test_ramp_settles_as_soon_as_the_output_is_on_the_live_reference():
    # A gap inside the settle tolerance is done on the first tick, not after duration_s.
    ramp = JointRamp([0], [1.0], duration_s=1.0, max_rate=0.5, now=0.0, settle_tol=0.02)
    qpos = np.array([1.01])
    assert ramp.apply(qpos, 0.0) == 1.0 and ramp.done and qpos[0] == 1.0
    # A small gap settles once the smoothstep target is within tolerance, well before 1 s.
    ramp = JointRamp([0], [0.0], duration_s=1.0, max_rate=0.5, now=0.0, settle_tol=0.02)
    t = 0.0
    while not ramp.done and t < 2.0:
        t += 0.01
        qpos[0] = 0.1
        ramp.apply(qpos, t)
    assert ramp.done and 0.3 < t < 0.9 and abs(qpos[0] - 0.1) <= 0.02
    # A large gap is bounded by the rate cap and ends after the blend clock.
    ramp = JointRamp([0], [0.0], duration_s=1.0, max_rate=0.5, now=0.0, settle_tol=0.02)
    t = 0.0
    while not ramp.done and t < 10.0:
        t += 0.01
        qpos[0] = 1.5
        ramp.apply(qpos, t)
    assert ramp.done and 2.9 < t < 3.1  # ~1.5 rad at 0.5 rad/s


def test_ramp_rate_cap_bounds_every_step_and_delays_settle():
    ramp = JointRamp([0], [0.0], duration_s=0.0, max_rate=0.5, now=0.0)  # no blend: pure rate limit
    qpos = np.array([2.0])
    outputs = []
    t = 0.0
    while not ramp.done and t < 10.0:
        t += 0.01
        qpos[0] = 2.0
        ramp.apply(qpos, t)
        outputs.append(qpos[0])
    steps = np.diff([0.0] + outputs)
    assert np.all(steps <= 0.5 * 0.01 + 1e-9) and abs(outputs[-1] - 2.0) <= ramp.settle_tol
    assert 3.9 < t < 4.1  # 2 rad at 0.5 rad/s


def test_ramp_follows_a_moving_reference_and_clamps_large_dt():
    ramp = JointRamp([0], [0.0], duration_s=1.0, max_rate=10.0, now=0.0)
    qpos = np.array([1.0])
    ramp.apply(qpos, 0.5)
    qpos[0] = 3.0  # the operator moved during the ramp
    ramp.apply(qpos, 5.0)  # dt is clamped to 0.1 s: at most 1 rad of travel this tick
    assert qpos[0] <= 0.5 + 1.0 + 1e-9 and not ramp.done


def test_rate_limiter_tracks_then_limits():
    limiter = RateLimiter(1.0)
    assert np.allclose(limiter.limit([0.0, 1.0], 0.0), [0.0, 1.0])  # first sample passes
    out = limiter.limit([1.0, 0.0], 0.1)
    assert np.allclose(out, [0.1, 0.9])
    limiter.track([5.0, 5.0], 0.2)  # pass-through resets the anchor
    assert np.allclose(limiter.limit([5.0, 5.05], 0.3), [5.0, 5.05])
    limiter.reset()
    assert np.allclose(limiter.limit([7.0, 7.0], 1.0), [7.0, 7.0])
