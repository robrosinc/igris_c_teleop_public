"""Python port of the igris_c_inference_public payload scanner.

Mirrors ``MotionReferenceSource.cpp`` (``FindKey``, ``ArrayBounds``,
``NumberArray``, ``StringArray``, ``RosMotionSource::ingest``) so a contract
test can assert what the C++ consumer would extract from a serialized frame.
Keep in step with that file; it is the only oracle for silent frame drops.
"""

from __future__ import annotations

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
