"""Joint-space ramps shared by the convergence gate and the arm hold release.

``JointRamp`` drives a subset of a MuJoCo ``qpos`` vector from a fixed start
(the robot's measured joints, or a held arm pose) to the live retargeted
reference: ``target = start + s(t) * (live - start)`` with a smoothstep ``s``
over ``duration_s``, then a per-joint rate cap so the output never moves
faster than ``max_rate`` even when the live reference jumps. The live
reference keeps moving during the ramp, so the ramp ends exactly on it.

``RateLimiter`` is the scalar-vector form used for the normalized hand command.
"""

from __future__ import annotations

import numpy as np


def smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, float(u)))
    return u * u * (3.0 - 2.0 * u)


class JointRamp:
    """Blend ``qpos[indices]`` from ``start`` to the live values it finds in ``qpos``."""

    def __init__(
        self,
        indices,
        start,
        *,
        duration_s: float,
        max_rate: float,
        now: float,
        settle_tol: float = 0.02,
        max_dt: float = 0.1,
    ) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self._out = np.array(start, dtype=np.float64).reshape(-1)
        if self._out.shape != self.indices.shape:
            raise ValueError(f"start has {self._out.size} values for {self.indices.size} joints.")
        self._start = self._out.copy()
        self.duration_s = max(0.0, float(duration_s))
        self.max_rate = float(max_rate)
        self.settle_tol = float(settle_tol)
        self.max_dt = float(max_dt)
        self._t0 = float(now)
        self._last = float(now)
        self.progress = 0.0
        self.done = False
        self.residual = 0.0

    @property
    def output(self) -> np.ndarray:
        return self._out.copy()

    def apply(self, qpos: np.ndarray, now: float) -> float:
        """Overwrite ``qpos[indices]`` with the ramped values; return the progress in [0, 1]."""

        now = float(now)
        dt = min(self.max_dt, max(0.0, now - self._last))
        self._last = now
        live = np.asarray(qpos[self.indices], dtype=np.float64)
        u = 1.0 if self.duration_s <= 0.0 else (now - self._t0) / self.duration_s
        s = smoothstep(u)
        target = self._start + s * (live - self._start)
        step = target - self._out
        if self.max_rate > 0.0:
            limit = self.max_rate * dt
            step = np.clip(step, -limit, limit)
        self._out = self._out + step
        qpos[self.indices] = self._out
        self.residual = float(np.max(np.abs(self._out - live))) if self._out.size else 0.0
        if s >= 1.0 and self.residual <= self.settle_tol:
            self.done = True
            self.progress = 1.0
        else:
            self.progress = s
        return self.progress


class RateLimiter:
    """Per-element rate cap on a vector command (used on the hand command while converging)."""

    def __init__(self, max_rate: float, *, max_dt: float = 0.1) -> None:
        self.max_rate = float(max_rate)
        self.max_dt = float(max_dt)
        self._prev: np.ndarray | None = None
        self._last: float | None = None

    def reset(self) -> None:
        self._prev = None
        self._last = None

    def track(self, values, now: float) -> None:
        """Record ``values`` as the current output without limiting (pass-through mode)."""

        self._prev = np.array(values, dtype=np.float64).reshape(-1)
        self._last = float(now)

    def limit(self, values, now: float) -> np.ndarray:
        target = np.array(values, dtype=np.float64).reshape(-1)
        if self._prev is None or self._last is None or self._prev.shape != target.shape:
            self.track(target, now)
            return target.copy()
        dt = min(self.max_dt, max(0.0, float(now) - self._last))
        self._last = float(now)
        if self.max_rate <= 0.0:
            self._prev = target.copy()
            return target.copy()
        step = np.clip(target - self._prev, -self.max_rate * dt, self.max_rate * dt)
        self._prev = self._prev + step
        return self._prev.copy()


__all__ = ["JointRamp", "RateLimiter", "smoothstep"]
