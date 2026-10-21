"""Convergence gate: start the reference at the robot's measured pose and ramp to the operator.

Without it the reference published to the inference node jumps to the
operator's pose the moment teleoperation (re)starts or the robot enters
low-level control, and the robot follows that jump (igris_c's own SlowStart
only rate-limits the arms, once, at mode entry). The gate replaces the 31
hinge joints of the retargeted ``qpos`` with a :class:`~.ramp.JointRamp` from
the latest ``LowState`` toward the live reference; everything downstream (FK
keybodies, finite-difference velocities, refstate overlay) is computed from
the ramped ``qpos``, so the inference node sees one consistent, slowly moving
reference. The root pose (``qpos[:7]``) is never touched. The ramp ends as
soon as every hinge is within ``settle_tol`` of the live reference (a
reference already next to the robot releases at once); ``duration_s`` only
shapes the soft start and ``max_rate`` bounds the speed of a large gap.

Triggers (each starts, or restarts, the ramp):

* the headset reports ``TeleopState.RESUMING`` after any other state (first
  Start and every Resume; the VRS1 ``teleop_state`` field, vr_link 2.1);
* the skeleton (re)connects while the headset sends no ``teleop_state``
  (pre-2.1 headsets; connection edge is the only start signal they give);
* the robot enters the low-level family (``ROBOT_STATE_LOW`` /
  ``ROBOT_STATE_WALK_LOW`` on the ``robotstate`` topic) from another mode,
  e.g. right B after a HOME preset while the headset stayed Active.

A trigger without a fresh ``LowState`` (older than ``lowstate_max_age_s`` or
none at all, e.g. ``--no_dds``) is logged and skipped: the reference passes
through unchanged, as before this module existed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from vr_link.types import TeleopState

from .ramp import JointRamp, RateLimiter

logger = logging.getLogger("igris_c_teleop.shaping.converge")

# igris_c_sdk RobotState_.state() values of the modes in which lowcmd drives the joints.
# Mirrors ROBOT_STATE_LOW / ROBOT_STATE_WALK_LOW of the pinned SDK (checked at start-up
# against the binding when available, see ``low_level_modes_from_sdk``).
DEFAULT_LOW_LEVEL_MODES: frozenset[int] = frozenset({3, 6})

TRIGGER_RESUME = "teleop_state RESUMING"
TRIGGER_CONNECT = "skeleton connected (no teleop_state from this headset)"
TRIGGER_LOW_LEVEL = "robot entered low-level control"


def low_level_modes_from_sdk(sdk) -> frozenset[int]:
    """The low-level mode set from the binding's constants, else the pinned defaults."""

    modes = set()
    for name in ("ROBOT_STATE_LOW", "ROBOT_STATE_WALK_LOW"):
        value = getattr(sdk, name, None)
        if value is not None:
            modes.add(int(value))
    return frozenset(modes) if modes else DEFAULT_LOW_LEVEL_MODES


@dataclass(frozen=True)
class ConvergeStatus:
    converging: bool
    progress: float  # smoothstep blend fraction 0..1 while converging, 1.0 when idle (informational only)
    trigger: str | None = None  # set on the tick a ramp started (log/feedback)
    skipped: str | None = None  # set when a trigger could not start (no fresh LowState)
    residual: float = 0.0  # max |ramp output - live reference| in rad while converging, 0.0 when idle


class ConvergeGate:
    def __init__(
        self,
        hinge_indices,
        *,
        duration_s: float = 1.0,
        max_rate: float = 0.5,
        hand_rate: float = 1.0,
        lowstate_max_age_s: float = 0.5,
        low_level_modes: frozenset[int] = DEFAULT_LOW_LEVEL_MODES,
        enabled: bool = True,
        settle_tol: float = 0.02,
    ) -> None:
        self.hinge_indices = np.asarray(hinge_indices, dtype=np.int64)
        self.duration_s = float(duration_s)
        self.max_rate = float(max_rate)
        self.lowstate_max_age_s = float(lowstate_max_age_s)
        self.low_level_modes = frozenset(int(m) for m in low_level_modes)
        self.enabled = bool(enabled)
        self.settle_tol = float(settle_tol)
        self._ramp: JointRamp | None = None
        self._hand = RateLimiter(hand_rate)
        self._prev_teleop_state: TeleopState | None = None
        self._prev_mode: int | None = None
        self.started = 0
        self.skipped = 0

    @property
    def active(self) -> bool:
        return self._ramp is not None

    @property
    def progress(self) -> float:
        return self._ramp.progress if self._ramp is not None else 1.0

    def reset(self) -> None:
        """Skeleton lost: drop the ramp and the headset-state memory (a reconnect in RESUMING triggers again)."""

        self._ramp = None
        self._prev_teleop_state = None
        self._hand.reset()

    def _triggers(self, teleop_state: TeleopState, connected_edge: bool, robot_mode: int | None) -> list[str]:
        found: list[str] = []
        if teleop_state == TeleopState.RESUMING and self._prev_teleop_state != TeleopState.RESUMING:
            found.append(TRIGGER_RESUME)
        elif connected_edge and teleop_state == TeleopState.UNKNOWN:
            found.append(TRIGGER_CONNECT)
        self._prev_teleop_state = teleop_state
        if robot_mode is not None:
            entering = robot_mode in self.low_level_modes and self._prev_mode is not None and self._prev_mode not in self.low_level_modes
            self._prev_mode = int(robot_mode)
            if entering:
                found.append(TRIGGER_LOW_LEVEL)
        return found

    def update(
        self,
        qpos: np.ndarray,
        *,
        now: float,
        teleop_state: TeleopState,
        connected_edge: bool,
        robot_mode: int | None,
        lowstate: tuple[np.ndarray, float] | None,
    ) -> ConvergeStatus:
        """Shape ``qpos`` in place for this tick; ``lowstate`` is ``(q31, age_s)`` or None."""

        triggers = self._triggers(teleop_state, connected_edge, robot_mode)
        if not self.enabled:
            return ConvergeStatus(False, 1.0)
        trigger = triggers[0] if triggers else None
        skipped = None
        if trigger is not None:
            if lowstate is None or lowstate[1] > self.lowstate_max_age_s:
                skipped = trigger
                self.skipped += 1
                logger.warning(
                    "convergence ramp NOT started (%s): no fresh LowState (%s); the reference passes through unchanged",
                    trigger,
                    "none received" if lowstate is None else f"age {lowstate[1]:.2f} s > {self.lowstate_max_age_s:.2f} s",
                )
                trigger = None
            else:
                q0 = np.asarray(lowstate[0], dtype=np.float64).reshape(-1)
                if q0.shape != self.hinge_indices.shape:
                    raise ValueError(f"LowState has {q0.size} joints, the reference has {self.hinge_indices.size}.")
                gap = float(np.max(np.abs(q0 - qpos[self.hinge_indices])))
                self._ramp = JointRamp(
                    self.hinge_indices, q0, duration_s=self.duration_s, max_rate=self.max_rate, now=now, settle_tol=self.settle_tol
                )
                self.started += 1
                logger.warning(
                    "convergence ramp started (%s): max gap %.3f rad, %.1f s smoothstep, %.2f rad/s cap", trigger, gap, self.duration_s, self.max_rate
                )
        if self._ramp is None:
            return ConvergeStatus(False, 1.0, trigger, skipped)
        progress = self._ramp.apply(qpos, now)
        if self._ramp.done:
            logger.warning("convergence ramp finished after %.2f s (residual %.3f rad)", now - self._ramp._t0, self._ramp.residual)
            self._ramp = None
            return ConvergeStatus(False, 1.0, trigger, skipped)
        return ConvergeStatus(True, progress, trigger, skipped, residual=self._ramp.residual)

    def shape_hand(self, command: np.ndarray | None, now: float) -> np.ndarray | None:
        """Rate-limit the 12-motor hand command while the ramp is active; pass through otherwise."""

        if command is None:
            return None
        if self._ramp is None:
            self._hand.track(command, now)
            return command
        return self._hand.limit(command, now)


__all__ = [
    "DEFAULT_LOW_LEVEL_MODES",
    "TRIGGER_CONNECT",
    "TRIGGER_LOW_LEVEL",
    "TRIGGER_RESUME",
    "ConvergeGate",
    "ConvergeStatus",
    "low_level_modes_from_sdk",
]
