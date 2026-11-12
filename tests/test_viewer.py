"""Opt-in reference viewer: IK target drawing, loop-safety of update(), CLI. No window is opened."""

from __future__ import annotations

import contextlib
import threading
from types import SimpleNamespace

import mujoco as mj
import numpy as np
import pytest
from conftest import ROBOT
from general_motion_retargeting import ROBOT_XML_DICT, GeneralMotionRetargeting
from vr_fixtures import make_state

from igris_c_teleop.app import producer
from igris_c_teleop.app.viewer import TARGET_BONES, ReferenceViewer, draw_targets, ik_target_names
from igris_c_teleop.uplink.vrs1_source import build_human_data

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
BONE_BODIES = {name for bone in TARGET_BONES for name in bone}


class FakeHandle:
    """The subset of mujoco.viewer.Handle the viewer uses, without GLFW."""

    def __init__(self, model: mj.MjModel, maxgeom: int = 200) -> None:
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()
        self.user_scn = mj.MjvScene(model, maxgeom=maxgeom)
        self.running = True
        self.synced = 0
        self.closed = 0
        self.fail_sync = False

    def lock(self):
        return contextlib.nullcontext()

    def is_running(self) -> bool:
        return self.running

    def sync(self) -> None:
        if self.fail_sync:
            raise RuntimeError("sync failed")
        self.synced += 1

    def close(self) -> None:
        self.closed += 1
        self.running = False


def _targets(names, z: float = 1.0):
    return {name: [np.array([0.1 * index, 0.0, z]), IDENTITY.copy()] for index, name in enumerate(names)}


def _viewer(fps: float = 30.0, names=tuple(sorted(BONE_BODIES))):
    handles: list[FakeHandle] = []

    def launch(model, data):
        handles.append(FakeHandle(model))
        return handles[-1]

    viewer = ReferenceViewer(str(ROBOT_XML_DICT[ROBOT]), base_body="base_link", target_names=names, fps=fps, launch=launch)
    return viewer, handles[0]


@pytest.fixture(scope="module")
def gmr():
    return GeneralMotionRetargeting(src_human="xrobot", tgt_robot=ROBOT, actual_human_height=1.65, verbose=False)


def test_ik_targets_come_from_the_gmr_instance(gmr):
    # Pins the GMR attributes the viewer reads (a submodule bump that renames them must fail here).
    names = ik_target_names(gmr)
    assert len(names) == len(set(names)) == 15
    assert set(names) == BONE_BODIES  # every bone endpoint is an IK target and every target is on the skeleton

    qpos = gmr.retarget(build_human_data(make_state()), offset_to_ground=True)
    assert qpos.shape == (gmr.model.nq,)
    for name in names:
        pos, quat = gmr.scaled_human_data[name]
        assert np.shape(pos) == (3,) and np.shape(quat) == (4,)
    assert mj.MjModel.from_xml_path(gmr.xml_file).nq == gmr.model.nq


def test_draw_targets_frames_and_bones(model):
    scn = mj.MjvScene(model, maxgeom=200)
    names = tuple(sorted(BONE_BODIES))
    targets = _targets(names)
    assert draw_targets(scn, targets, names) == 3 * len(names) + len(TARGET_BONES)
    assert scn.ngeom == 3 * len(names) + len(TARGET_BONES)

    # One frame = three arrows from the target position along the target's x, y, z axes.
    quat = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])  # +90 deg about z: x -> y, y -> -x
    scn.ngeom = 0
    assert draw_targets(scn, {"Pelvis": [np.array([1.0, 2.0, 3.0]), 2.0 * quat]}, ("Pelvis",)) == 3
    directions = [np.array(scn.geoms[i].mat).reshape(3, 3)[:, 2] for i in range(3)]
    np.testing.assert_allclose(directions, [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], atol=1e-6)
    for i in range(3):
        assert scn.geoms[i].type == mj.mjtGeom.mjGEOM_ARROW
        np.testing.assert_allclose(scn.geoms[i].pos, [1.0, 2.0, 3.0])


def test_draw_targets_skips_bad_input_and_respects_the_scene_size(model):
    scn = mj.MjvScene(model, maxgeom=200)
    targets = {
        "Pelvis": [np.array([0.0, 0.0, 1.0]), IDENTITY],
        "Spine3": [np.array([0.0, 0.0, 1.3]), np.zeros(4)],  # degenerate quaternion: bone only
        "Head": [np.array([0.0, np.nan, 1.6]), IDENTITY],  # not finite: dropped with its bone
    }
    # Left_Hip is not in the data at all.
    assert draw_targets(scn, targets, ("Pelvis", "Spine3", "Head", "Left_Hip")) == 3 + 1
    assert scn.geoms[3].type == mj.mjtGeom.mjGEOM_CAPSULE
    np.testing.assert_allclose(scn.geoms[3].pos, [0.0, 0.0, 1.15])

    names = tuple(sorted(BONE_BODIES))
    small = mj.MjvScene(model, maxgeom=10)
    assert draw_targets(small, _targets(names), names) == 10
    assert small.ngeom == 10


def test_update_shows_the_reference_and_decimates(keyframe_qpos):
    viewer, handle = _viewer(fps=30.0)
    assert viewer.active and handle.synced == 1
    assert handle.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] == 1 and handle.cam.distance == 3.0

    qpos = keyframe_qpos.copy()
    qpos[:3] = [0.4, -0.2, 0.9]
    names = tuple(sorted(BONE_BODIES))
    assert viewer.update(qpos, _targets(names), now=10.0)
    assert handle.synced == 2
    np.testing.assert_allclose(viewer._data.qpos, qpos)
    np.testing.assert_allclose(handle.cam.lookat, viewer._data.xpos[viewer._model.body("base_link").id])
    assert handle.user_scn.ngeom == 3 * len(names) + len(TARGET_BONES)

    # No targets: the previous overlay is cleared, the robot still follows.
    assert viewer.update(qpos, None, now=11.0)
    assert handle.user_scn.ngeom == 0

    # 100 Hz ticks for one second -> about fps updates, the other ticks return without touching the window.
    synced = handle.synced
    for tick in range(100):
        assert viewer.update(qpos, None, now=20.0 + 0.01 * tick)
    assert 29 <= handle.synced - synced <= 31

    count, mean_s, max_s = viewer.take_stats()
    assert count == viewer.updates == handle.synced - 1 and 0.0 < mean_s <= max_s
    assert viewer.take_stats() == (0, 0.0, 0.0)


def test_closed_window_turns_the_viewer_off_without_raising(keyframe_qpos):
    viewer, handle = _viewer()
    handle.running = False  # operator closed the window
    assert viewer.update(keyframe_qpos, None, now=1.0) is False
    assert not viewer.active and handle.closed == 1 and handle.synced == 1
    assert viewer.update(keyframe_qpos, None, now=2.0) is False
    viewer.close()
    viewer.close()
    assert handle.closed == 1


@pytest.mark.parametrize("failure", ["sync", "qpos_size"])
def test_update_error_turns_the_viewer_off_without_raising(keyframe_qpos, failure):
    viewer, handle = _viewer()
    handle.fail_sync = failure == "sync"
    qpos = keyframe_qpos if failure == "sync" else keyframe_qpos[:-1]
    assert viewer.update(qpos, None, now=1.0) is False
    assert not viewer.active and handle.closed == 1
    assert viewer.update(keyframe_qpos, None, now=2.0) is False


def test_close_waits_for_the_render_thread():
    stop = threading.Event()
    started: list[threading.Thread] = []

    def launch(model, data):
        handle = FakeHandle(model)
        handle.close = stop.set  # the real Handle.close() only requests the exit
        thread = threading.Thread(target=stop.wait, daemon=True)
        thread.start()
        started.append(thread)
        return handle

    viewer = ReferenceViewer(str(ROBOT_XML_DICT[ROBOT]), base_body="base_link", target_names=(), launch=launch)
    assert started[0].is_alive()
    viewer.close()
    assert not started[0].is_alive() and not viewer.active


def test_open_is_skipped_without_a_display(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert ReferenceViewer.open(SimpleNamespace(), ROBOT) is None  # headless host: no window, no error
    with pytest.raises(ValueError, match="fps"):
        ReferenceViewer(str(ROBOT_XML_DICT[ROBOT]), base_body="base_link", target_names=(), fps=0.0, launch=lambda m, d: None)


def test_viewer_is_on_by_default_on_the_cli():
    args = producer.parse_args([])
    assert not args.no_viewer and args.viewer_fps == 30.0
    assert producer.parse_args(["--no_viewer"]).no_viewer
    assert producer.parse_args(["--viewer_fps", "15"]).viewer_fps == 15.0
    with pytest.raises(SystemExit, match="viewer_fps"):
        producer.main(["--no_dds", "--viewer_fps", "0"])
