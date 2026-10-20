"""Normalized 12-motor finger command (``handcmd`` motor order) from VRS1 hand data.

Wraps gmr's ``igris_c_hand_command.HandCommandState``: 0 = open, 1 = closed;
motor order 11..16 right hand, 21..26 left hand (thumb flexion, index, middle,
ring, little, thumb CMC).
"""

from __future__ import annotations

import numpy as np
from general_motion_retargeting.hand_retarget.igris_c_hand_command import (
    DEFAULT_CALIBRATION,
    HAND_COMMAND_DIM,
    MOTOR_IDS,
    SIDE_MOTOR_SLOTS,
    FingerCurlCalibration,
    HandCommandState,
    hand_keypoints_to_normalized,
)
from general_motion_retargeting.hand_retarget.igris_c_hand_command import (
    GRIP_FULL_COMMAND as GMR_GRIP_FULL_COMMAND,
)
from vr_link.types import VRInputState

FINGER_SOURCES = ("auto", "hand", "grip", "none")

# Motor command at a fully squeezed controller grip. gmr's HandCommandState caps the
# grip source at GMR_GRIP_FULL_COMMAND (0.6, a 2026-09-15 sim choice); the grip
# block is rescaled here so the published handcmd spans the full motor range.
GRIP_FULL_COMMAND = 1.0


class HandTargets:
    """Normalized 12-motor finger command selected by ``--finger_source``.

    Per hand, ``auto`` arbitrates between the hand-tracking keypoints (finger
    curl) and the controller grip and holds when neither is conclusive;
    ``hand``/``grip`` force one source (a forced ``hand`` holds while
    untracked); ``none`` disables the hand command. Hands are independent, the
    initial command is the open hand, and a hand that cannot be updated keeps
    its last values. A grip-driven hand is linear in the grip: released -> 0,
    fully squeezed -> ``grip_full_command``. ``update`` returns the command in
    ``MOTOR_IDS`` order (values in [0, 1]) or ``None`` for ``none``.
    """

    def __init__(
        self,
        finger_source: str,
        *,
        calibration: FingerCurlCalibration = DEFAULT_CALIBRATION,
        curl_fn=hand_keypoints_to_normalized,
        grip_full_command: float = GRIP_FULL_COMMAND,
    ) -> None:
        if finger_source not in FINGER_SOURCES:
            raise ValueError(f"finger_source must be one of {FINGER_SOURCES}, got {finger_source!r}.")
        if not 0.0 < float(grip_full_command) <= 1.0:
            raise ValueError(f"grip_full_command must be in (0, 1], got {grip_full_command!r}.")
        self.source = finger_source
        self.motor_ids = MOTOR_IDS
        self.grip_full_command = float(grip_full_command)
        # gmr writes grip * GMR_GRIP_FULL_COMMAND; undo that and apply ours.
        self._grip_rescale = self.grip_full_command / GMR_GRIP_FULL_COMMAND
        self._command = np.zeros(HAND_COMMAND_DIM, dtype=np.float64)
        self._state: HandCommandState | None = None
        if finger_source != "none":
            self._state = HandCommandState(
                None if finger_source == "auto" else finger_source,
                calibration=calibration,
                curl_fn=curl_fn,
            )

    @property
    def enabled(self) -> bool:
        return self._state is not None

    @property
    def tracked(self) -> tuple[bool, bool]:
        """(left, right): whether fresh hand-tracking keypoints produced this frame's command."""

        return self._state.fresh_tracking if self._state is not None else (False, False)

    @property
    def active_sources(self) -> tuple[str, str]:
        """(left, right): hand / grip / hold decided for the latest frame."""

        return self._state.active_sources if self._state is not None else ("hold", "hold")

    def reset(self) -> None:
        """Skeleton stream lost: forget the source decisions, keep the last command (hold)."""

        if self._state is not None:
            self._state.reset()

    def update(self, state: VRInputState) -> np.ndarray | None:
        if self._state is None:
            return None
        decisions = self._state.update(state.left_hand, state.right_hand, state.left_input, state.right_input)
        raw = self._state.command
        for side, decision in decisions.items():
            block = SIDE_MOTOR_SLOTS[side]
            if decision == "hand":
                self._command[block] = raw[block]
            elif decision == "grip":
                self._command[block] = np.clip(raw[block] * self._grip_rescale, 0.0, 1.0)
            # "hold": keep this side's last published block.
        return self._command.copy()


__all__ = ["FINGER_SOURCES", "GRIP_FULL_COMMAND", "HAND_COMMAND_DIM", "MOTOR_IDS", "HandTargets"]
