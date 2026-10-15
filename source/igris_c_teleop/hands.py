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
    FingerCurlCalibration,
    HandCommandState,
    hand_keypoints_to_normalized,
)
from vr_link.types import VRInputState

FINGER_SOURCES = ("auto", "hand", "grip", "none")


class HandTargets:
    """Normalized 12-motor finger command selected by ``--finger_source``.

    Per hand, ``auto`` arbitrates between the hand-tracking keypoints (finger
    curl) and the controller grip and holds when neither is conclusive;
    ``hand``/``grip`` force one source (a forced ``hand`` holds while
    untracked); ``none`` disables the hand command. Hands are independent, the
    initial command is the open hand, and a hand that cannot be updated keeps
    its last values. ``update`` returns the command in ``MOTOR_IDS`` order
    (values in [0, 1]) or ``None`` for ``none``.
    """

    def __init__(
        self,
        finger_source: str,
        *,
        calibration: FingerCurlCalibration = DEFAULT_CALIBRATION,
        curl_fn=hand_keypoints_to_normalized,
    ) -> None:
        if finger_source not in FINGER_SOURCES:
            raise ValueError(f"finger_source must be one of {FINGER_SOURCES}, got {finger_source!r}.")
        self.source = finger_source
        self.motor_ids = MOTOR_IDS
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
        self._state.update(state.left_hand, state.right_hand, state.left_input, state.right_input)
        return self._state.command


__all__ = ["FINGER_SOURCES", "HAND_COMMAND_DIM", "MOTOR_IDS", "HandTargets"]
