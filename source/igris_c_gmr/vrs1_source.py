"""VRS1 input sources: the live headset device and a ``.vrs1`` file replay."""

from __future__ import annotations

import struct
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from general_motion_retargeting.vr_uplink_convention import to_xrobot_convention
from vr_link.tcp_device import (
    VRS1_HEADER_FORMAT,
    VRS1_HEADER_SIZE,
    VRS1_MAGIC,
    VRS1_MAX_PAYLOAD_BYTES,
    TcpVRDevice,
    decode_vr_state_payload,
)
from vr_link.types import BODY_JOINT_NAMES, VRInputState

DEFAULT_POSE_PORT = 9102


def iter_vrs1_payloads(data: bytes) -> Iterator[bytes]:
    """Yield the FlatBuffers payload of every complete VRS1 record in ``data``.

    Resynchronizes on the magic like the TCP receiver; a trailing partial
    record is ignored.
    """

    magic = struct.pack(">I", VRS1_MAGIC)
    offset = 0
    while True:
        start = data.find(magic, offset)
        if start < 0 or start + VRS1_HEADER_SIZE > len(data):
            return
        _magic, size = struct.unpack(VRS1_HEADER_FORMAT, data[start : start + VRS1_HEADER_SIZE])
        if size == 0 or size > VRS1_MAX_PAYLOAD_BYTES:
            offset = start + len(magic)
            continue
        end = start + VRS1_HEADER_SIZE + size
        if end > len(data):
            return
        yield data[start + VRS1_HEADER_SIZE : end]
        offset = end


class Vrs1FileSource:
    """Replays a ``.vrs1`` recording with the ``TcpVRDevice`` polling interface.

    Each ``advance`` call yields the next decodable record (looping at the
    end), so the replay runs at the caller's tick rate rather than the recorded
    timing. Body joints go through the same gmr convention adapter as the live
    device.
    """

    def __init__(self, path: Path) -> None:
        self._payloads = list(iter_vrs1_payloads(Path(path).read_bytes()))
        if not self._payloads:
            raise ValueError(f"No complete VRS1 records in {path}.")
        self._index = 0
        self._connected = False

    def start(self) -> None:
        self._connected = True

    def close(self) -> None:
        self._connected = False

    @property
    def receiving(self) -> bool:
        return self._connected

    def advance(self, dt: float) -> VRInputState:
        del dt
        now = time.monotonic()
        if not self._connected:
            return VRInputState(timestamp=now, connected=False)
        for _ in range(len(self._payloads)):
            payload = self._payloads[self._index]
            self._index = (self._index + 1) % len(self._payloads)
            state = decode_vr_state_payload(payload, timestamp=now, body_joint_adapter=to_xrobot_convention)
            if state is not None:
                return state
        return VRInputState(timestamp=now, connected=False)

    def __len__(self) -> int:
        return len(self._payloads)


def open_live_device(host: str, port: int, stale_timeout: float) -> TcpVRDevice:
    """VRS1 TCP server the headset connects to, with the gmr joint convention applied."""

    return TcpVRDevice(host=host, port=port, stale_timeout=stale_timeout, body_joint_adapter=to_xrobot_convention)


def build_human_data(state: VRInputState) -> dict[str, list[np.ndarray]]:
    """GMR ``xrobot`` input: joint name -> [pos_xyz (m), quat_wxyz]."""

    return {
        name: [np.asarray(pose.position_m, dtype=np.float64), np.asarray(pose.quat_wxyz, dtype=np.float64)]
        for name, pose in zip(BODY_JOINT_NAMES, state.body_joints)
    }


def skeleton_complete(state: VRInputState) -> bool:
    """True when the record is connected and carries the full 24-joint skeleton."""

    return bool(state.connected) and len(state.body_joints) == len(BODY_JOINT_NAMES)


__all__ = [
    "DEFAULT_POSE_PORT",
    "Vrs1FileSource",
    "build_human_data",
    "iter_vrs1_payloads",
    "open_live_device",
    "skeleton_complete",
]
