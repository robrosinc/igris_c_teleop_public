"""Convergence gate: start the reference at the robot's measured pose and ramp to the operator.

Without it the reference published to the inference node jumps to the
operator's pose the moment teleoperation (re)starts or the robot enters
low-level control, and the robot follows that jump (igris_c's own SlowStart
only rate-limits the arms, once, at mode entry). The gate replaces the 31
hinge joints of the retargeted ``qpos`` with a :class:`~.ramp.JointRamp` from
the latest ``LowState`` toward the live reference; everything downstream (FK
keybodies, finite-difference velocities, refstate overlay) is computed from
the ramped ``qpos``, so the inference node sees one consistent, slowly moving
reference. The ramp never touches the root pose (``qpos[:7]``). The ramp ends as
soon as every hinge is within ``settle_tol`` of the live reference (a
reference already next to the robot releases at once); ``duration_s`` only
shapes the soft start and ``max_rate`` bounds the speed of a large gap.

Triggers (each starts, or restarts, the ramp):

* the headset reports ``TeleopState.RESUMING`` after any other state, or with
  a new ``resume_count`` (first Start and every Resume; the VRS1
  ``teleop_state`` / ``resume_count`` fields);
* the skeleton (re)connects while the headset sends no ``teleop_state``
  (pre-2.1 headsets; connection edge is the only start signal they give);
* the robot enters the low-level family (``ROBOT_STATE_LOW`` /
  ``ROBOT_STATE_WALK_LOW`` on the ``robotstate`` topic) from another mode,
  e.g. right B after a HOME preset while the headset stayed Active.

A ramp always starts from a fresh ``LowState``. A trigger without one (older
than ``lowstate_max_age_s`` or none received) waits: the gate holds the last
reference it returned (whole ``qpos``, root included, and the last hand
command) and reports ``converging``; with no earlier reference to hold it
asks the caller to publish nothing (``ConvergeStatus.publish``). The ramp
starts on the first tick a fresh ``LowState`` is there.

Resume acknowledgement: the headset stays in Resuming until this gate reports
the ``resume_count`` of that request back (``ConvergeStatus.converged_resume_count``
-> ``teleop/motion_feedback`` -> ``Signals.converged_resume_count``). The gate
does not count; it echoes the id of the RESUMING it saw start once no ramp runs
and no trigger waits while the headset still reports that id in RESUMING. A
disabled gate (``--no_converge``) echoes at once. The echo is cleared when the
skeleton is lost.
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


WAITING_LOG_PERIOD_S = 2.0


@dataclass(frozen=True)
class ConvergeStatus:
    converging: bool  # the reference is not the live one: the ramp runs, or a trigger waits for a fresh LowState
    progress: float  # smoothstep blend fraction 0..1 while converging, 1.0 when idle (informational only)
    trigger: str | None = None  # set on the tick a ramp started (log/feedback; velocity history must be dropped)
    deferred: str | None = None  # set on the tick a trigger began waiting for a fresh LowState
    residual: float = 0.0  # max |ramp output - live reference| in rad while the ramp runs, 0.0 otherwise
    waiting: bool = False  # a trigger waits for a fresh LowState; qpos holds the last returned reference
    publish: bool = True  # False only while waiting with no earlier reference to hold: nothing may go out
    converged_resume_count: int = 0  # echo of VRState.resume_count whose convergence finished, 0 = none


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
        self._prev_resume_count = 0
        self._pending: str | None = None  # trigger waiting for a fresh LowState
        self._pending_since = 0.0
        self._pending_logged: float | None = None  # time of the last "waiting" log, None before the first one
        self._held_qpos: np.ndarray | None = None  # last reference returned, held while a trigger waits
        self._resume_request = 0  # resume_count of the RESUMING last seen to start (0: none / headset without the field)
        self._converged_resume_count = 0
        self.started = 0
        self.deferred = 0

    @property
    def active(self) -> bool:
        return self._ramp is not None

    @property
    def waiting(self) -> bool:
        return self._pending is not None

    @property
    def progress(self) -> float:
        return self._ramp.progress if self._ramp is not None else 1.0

    def reset(self) -> None:
        """Skeleton lost: drop the ramp, a waiting trigger, the held reference, the headset-state memory and the echo.

        A reconnect in RESUMING triggers again; the echo restarts at 0 so a headset that
        restarted cannot read the id of an earlier session as its own.
        """

        self._ramp = None
        self._pending = None
        self._held_qpos = None
        self._prev_teleop_state = None
        self._prev_resume_count = 0
        self._resume_request = 0
        self._converged_resume_count = 0
        self._hand.reset()

    def _triggers(self, teleop_state: TeleopState, connected_edge: bool, robot_mode: int | None, resume_count: int) -> list[str]:
        found: list[str] = []
        resuming = teleop_state == TeleopState.RESUMING
        # A new id inside RESUMING is a new request whose PAUSED records the newest-only receiver never saw.
        if resuming and (self._prev_teleop_state != TeleopState.RESUMING or resume_count != self._prev_resume_count):
            found.append(TRIGGER_RESUME)
            self._resume_request = resume_count
        elif connected_edge and teleop_state == TeleopState.UNKNOWN:
            found.append(TRIGGER_CONNECT)
        self._prev_teleop_state = teleop_state
        self._prev_resume_count = resume_count
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
        resume_count: int = 0,
    ) -> ConvergeStatus:
        """Shape ``qpos`` in place for this tick; ``lowstate`` is ``(q31, age_s)`` or None.

        ``resume_count`` is the headset's ``VRState.resume_count`` of this record (0 from
        a headset without the field).
        """

        resume_count = int(resume_count)
        triggers = self._triggers(teleop_state, connected_edge, robot_mode, resume_count)
        if not self.enabled:
            # No ramp to wait for: acknowledge the request at once so the headset can leave Resuming.
            self._acknowledge(teleop_state, resume_count)
            return ConvergeStatus(False, 1.0, converged_resume_count=self._converged_resume_count)

        deferred = None
        if triggers:
            # The newest trigger replaces a running ramp or an older waiting trigger.
            if self._pending is None:
                self._pending_since = now
                self._pending_logged = None
            self._pending = triggers[0]
            self._ramp = None

        trigger = None
        if self._pending is not None:
            if lowstate is None or lowstate[1] > self.lowstate_max_age_s:
                reason = "none received" if lowstate is None else f"age {lowstate[1]:.2f} s > {self.lowstate_max_age_s:.2f} s"
                publish = self._held_qpos is not None
                if self._pending_logged is None:
                    deferred = self._pending
                    self.deferred += 1
                if self._pending_logged is None or now - self._pending_logged >= WAITING_LOG_PERIOD_S:
                    self._pending_logged = now
                    logger.warning(
                        "convergence ramp waiting (%s) for a fresh LowState (%s) since %.1f s; %s",
                        self._pending,
                        reason,
                        now - self._pending_since,
                        "the last reference is held" if publish else "nothing is published (no earlier reference to hold)",
                    )
                if publish:
                    qpos[:] = self._held_qpos
                return ConvergeStatus(
                    True, 0.0, None, deferred, waiting=True, publish=publish, converged_resume_count=self._converged_resume_count
                )
            q0 = np.asarray(lowstate[0], dtype=np.float64).reshape(-1)
            if q0.shape != self.hinge_indices.shape:
                raise ValueError(f"LowState has {q0.size} joints, the reference has {self.hinge_indices.size}.")
            gap = float(np.max(np.abs(q0 - qpos[self.hinge_indices])))
            self._ramp = JointRamp(
                self.hinge_indices, q0, duration_s=self.duration_s, max_rate=self.max_rate, now=now, settle_tol=self.settle_tol
            )
            trigger, self._pending = self._pending, None
            self.started += 1
            logger.warning(
                "convergence ramp started (%s): max gap %.3f rad, %.1f s smoothstep, %.2f rad/s cap", trigger, gap, self.duration_s, self.max_rate
            )

        converging, progress, residual = False, 1.0, 0.0
        if self._ramp is not None:
            progress = self._ramp.apply(qpos, now)
            if self._ramp.done:
                logger.warning("convergence ramp finished after %.2f s (residual %.3f rad)", now - self._ramp._t0, self._ramp.residual)
                self._ramp = None
                progress = 1.0
            else:
                converging, residual = True, self._ramp.residual
        self._held_qpos = np.array(qpos, dtype=np.float64)
        self._acknowledge(teleop_state, resume_count)
        return ConvergeStatus(
            converging, progress, trigger, None, residual=residual, converged_resume_count=self._converged_resume_count
        )

    def _acknowledge(self, teleop_state: TeleopState, resume_count: int) -> None:
        """Echo the id of the RESUMING this gate saw start once nothing shapes or holds the reference any more."""

        if resume_count == 0 or teleop_state != TeleopState.RESUMING or resume_count != self._resume_request:
            return
        if self._ramp is not None or self._pending is not None:
            return
        if self._converged_resume_count != resume_count:
            logger.info("resume %d acknowledged: the reference follows the operator", resume_count)
        self._converged_resume_count = resume_count

    def shape_hand(self, command: np.ndarray | None, now: float) -> np.ndarray | None:
        """Rate-limit the 12-motor hand command while the ramp is active; pass through otherwise.

        While a trigger waits for a fresh LowState the last command is held (None, i.e.
        nothing to publish, before the first one).
        """

        if command is None:
            return None
        if self._pending is not None:
            held = self._hand.current
            if held is not None:
                self._hand.track(held, now)  # keep the limiter clock running: the first ramp step stays bounded
            return held
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
