"""Python port of the igris_c_inference_public payload scanner.

Mirrors ``MotionReferenceSource.cpp`` (``FindKey``, ``ArrayBounds``,
``NumberArray``, ``StringArray``, ``RosMotionSource::ingest``) so a contract
test can assert what the C++ consumer would extract from a serialized frame.
Keep in step with that file; it is the only oracle for silent frame drops.

``source_stamp``, ``stamp_accepts`` and ``stamp_timeline_changed`` mirror the
frame identity rules of igris_c_inference_public ``feature/gmt_release_dev``
(c6c02f4): ``SourceStamp`` in ``MotionReferenceSource.cpp`` and
``MotionReferenceHistory::accepts`` / ``timelineChanged``.
"""

from __future__ import annotations

import json

NUMERIC = set("0123456789-+.eE")


def _identifier(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def find_key(payload: str, key: str) -> int:
    position = payload.find(key)
    while position != -1:
        before = position == 0 or not _identifier(payload[position - 1])
        after_position = position + len(key)
        after = after_position == len(payload) or not _identifier(payload[after_position])
        if before and after:
            return position
        position = payload.find(key, position + 1)
    return -1


def array_bounds(payload: str, key: str) -> tuple[int, int] | None:
    key_position = find_key(payload, key)
    if key_position < 0:
        return None
    begin = payload.find("[", key_position)
    if begin < 0:
        return None
    depth = 0
    for i in range(begin, len(payload)):
        if payload[i] == "[":
            depth += 1
        elif payload[i] == "]":
            depth -= 1
            if depth == 0:
                return begin, i
    return None


def number_array(payload: str, key: str) -> list[float] | None:
    bounds = array_bounds(payload, key)
    if bounds is None:
        return None
    begin, end = bounds
    out: list[float] = []
    token = ""
    for ch in payload[begin + 1 : end]:
        if ch in NUMERIC:
            token += ch
        elif token:
            out.append(float(token))
            token = ""
    if token:
        out.append(float(token))
    return out or None


def string_array(payload: str, key: str) -> list[str] | None:
    bounds = array_bounds(payload, key)
    if bounds is None:
        return None
    begin, end = bounds
    out: list[str] = []
    quoted = False
    quote = ""
    token = ""
    for ch in payload[begin + 1 : end]:
        if not quoted and ch in ("'", '"'):
            quoted, quote = True, ch
        elif quoted and ch == quote:
            out.append(token)
            token = ""
            quoted = False
        elif quoted:
            token += ch
    return out or None


def ingest(payload: str, joint_count: int) -> dict[str, object] | None:
    """Return the parsed frame the consumer would accept, or None when it drops it."""

    q = number_array(payload, "dof_pos")
    qd = number_array(payload, "dof_vel")
    root = number_array(payload, "root_rot")
    bodies = string_array(payload, "link_body_list")
    local = number_array(payload, "keybody_rot_local")
    if (
        q is None
        or qd is None
        or root is None
        or len(root) < 4
        or bodies is None
        or local is None
        or len(q) != joint_count
        or len(qd) != joint_count
        or len(local) != len(bodies) * 4
    ):
        return None
    frame: dict[str, object] = {"dof_pos": q, "dof_vel": qd, "root_rot": root[:4], "link_body_list": bodies, "keybody_rot_local": local}
    for optional in ("keybody_pos_local", "root_pos", "root_vel", "root_angvel"):
        values = number_array(payload, optional)
        if values is not None:
            frame[optional] = values
    return frame


LEGACY_SESSION = "legacy-public-teleop"
MAX_UNIX_MS = 8000000000000
INT64_MAX = 2**63 - 1


def source_stamp(payload: str) -> dict[str, object] | None:
    """The frame identity the consumer derives from ``record_meta``.

    None when the frame carries no usable stamp; ValueError where the C++ throws.
    A live ROS source drops the frame in both cases.
    """

    key = find_key(payload, "record_meta")
    if key < 0:
        return None
    colon = payload.find(":", key)
    if colon < 0:
        return None
    begin = colon + 1
    while begin < len(payload) and payload[begin] in " \t\r\n":
        begin += 1
    if begin >= len(payload) or payload[begin] != "{":
        return None
    depth = 0
    quote = ""
    escaped = False
    for end in range(begin, len(payload)):
        ch = payload[end]
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                meta = json.loads(payload[begin : end + 1])
                if "frame_idx" not in meta or "unix_ms" not in meta:
                    return None
                has_session = "session_id" in meta
                session = str(meta["session_id"]) if has_session else LEGACY_SESSION
                unix_ms = int(meta["unix_ms"])
                index = int(meta["frame_idx"])
                if not session or index < 0 or unix_ms <= 0 or unix_ms > MAX_UNIX_MS:
                    raise ValueError("invalid motion reference source timestamp")
                stamp: dict[str, object] = {
                    "session": session,
                    "frame_index": index if has_session else unix_ms,
                    "unix_ms": unix_ms,
                    "source_timestamp_ns": None,
                }
                if "source_timestamp_ns" in meta:
                    source_ns = int(meta["source_timestamp_ns"])
                    if source_ns <= 0 or source_ns > INT64_MAX or not meta.get("source_session_id"):
                        raise ValueError("invalid motion reference device timestamp/session")
                    stamp["source_timestamp_ns"] = source_ns
                    stamp["session"] = session + ":" + str(meta["source_session_id"])
                return stamp
    raise ValueError("unterminated motion reference record_meta")


def stamp_accepts(previous: dict[str, object] | None, stamp: dict[str, object]) -> bool:
    """False for a frame the consumer treats as a repeat of ``previous`` (it does not count as new input)."""

    if previous is None or stamp["session"] != previous["session"]:
        return True
    if stamp["frame_index"] <= previous["frame_index"]:
        return False
    if stamp["source_timestamp_ns"] is None or previous["source_timestamp_ns"] is None:
        return True
    return stamp["source_timestamp_ns"] > previous["source_timestamp_ns"]


def stamp_timeline_changed(previous: dict[str, object], stamp: dict[str, object]) -> bool:
    """True when the consumer drops its history and re-aligns the heading on this frame."""

    if stamp["session"] != previous["session"]:
        return True
    if (stamp["source_timestamp_ns"] is None) != (previous["source_timestamp_ns"] is None):
        return True

    def source_ns(item: dict[str, object]) -> int:
        return item["source_timestamp_ns"] if item["source_timestamp_ns"] is not None else item["unix_ms"] * 1000000

    return source_ns(stamp) < source_ns(previous)
