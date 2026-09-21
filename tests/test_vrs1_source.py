"""VRS1 record framing and .vrs1 file replay."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
from general_motion_retargeting.vr_uplink_convention import to_xrobot_convention
from vr_link.tcp_device import VRS1_HEADER_SIZE
from vr_link.types import BODY_JOINT_NAMES

from igris_c_gmr import vrs1_source
from igris_c_gmr.neck import head_relative_neck_angles
from vr_fixtures import IDENTITY, vrs1_record

# Raw headset captures are not committed; point IGRIS_C_GMR_VRS1_DIR at a folder of .vrs1 files
# (e.g. hit-bench outputs/hand_recordings) or drop them into tests/data/.
RECORDING_DIR = Path(os.environ.get("IGRIS_C_GMR_VRS1_DIR", Path(__file__).resolve().parent / "data"))


def test_iter_vrs1_payloads_resyncs_on_garbage():
    records = [vrs1_record(0.1), vrs1_record(0.2), vrs1_record(0.3)]
    blob = b"junk" + records[0] + b"\x00\x01" + records[1] + records[2][: VRS1_HEADER_SIZE + 5]  # truncated tail
    payloads = list(vrs1_source.iter_vrs1_payloads(blob))
    assert payloads == [records[0][VRS1_HEADER_SIZE:], records[1][VRS1_HEADER_SIZE:]]


def test_vrs1_file_source_loops_and_applies_convention(tmp_path):
    path = tmp_path / "two.vrs1"
    path.write_bytes(vrs1_record(0.1) + vrs1_record(0.2))
    source = vrs1_source.Vrs1FileSource(path)
    assert len(source) == 2
    assert not source.advance(0.0).connected  # not started
    source.start()
    first, second, third = (source.advance(0.0) for _ in range(3))
    assert [s.right_input.grip for s in (first, second, third)] == pytest.approx([0.1, 0.2, 0.1])
    assert first.body_joints[0].quat_wxyz == pytest.approx(to_xrobot_convention("Pelvis", IDENTITY))
    assert first.head.quat_wxyz == pytest.approx(IDENTITY)
    assert vrs1_source.skeleton_complete(first)
    assert set(vrs1_source.build_human_data(first)) == set(BODY_JOINT_NAMES)
    source.close()
    assert not source.advance(0.0).connected
    assert not vrs1_source.skeleton_complete(source.advance(0.0))


def test_vrs1_file_source_rejects_empty(tmp_path):
    path = tmp_path / "empty.vrs1"
    path.write_bytes(b"")
    with pytest.raises(ValueError):
        vrs1_source.Vrs1FileSource(path)


@pytest.mark.parametrize("name", sorted(p.name for p in RECORDING_DIR.glob("*.vrs1")) or ["<none>"])
def test_recorded_vrs1_decodes_with_neck_angles(name):
    path = RECORDING_DIR / name
    if not path.is_file():
        pytest.skip(f"no .vrs1 recordings in {RECORDING_DIR} (set IGRIS_C_GMR_VRS1_DIR)")
    source = vrs1_source.Vrs1FileSource(path)
    source.start()
    state = source.advance(0.0)
    assert vrs1_source.skeleton_complete(state)
    angles = head_relative_neck_angles(state)
    if angles is not None:
        assert all(math.isfinite(a) and abs(a) < math.pi for a in angles)
