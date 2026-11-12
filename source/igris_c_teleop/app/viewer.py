"""In-process MuJoCo window for the producer (on by default, ``--no_viewer``): reference pose and GMR IK targets.

Shows, on the operator PC, what the producer is publishing:

  * the robot at the shaped reference ``qpos`` (after the neck override, arm hold and the convergence
    ramp), i.e. the pose that goes out on ``/gmr/teleop/retarget_frame``;
  * the GMR IK targets of the same tick (``GeneralMotionRetargeting.scaled_human_data``: the scaled,
    offset and ground-aligned human bodies the solver pulls the robot links to) as RGB frames joined
    by skeleton bones.

Display only: nothing here feeds back into the reference. It runs inside the 100 Hz loop, so the cost
is kept bounded: updates are decimated to ``fps``, the producer calls ``update`` after every output of
the tick has been published, and a closed window or the first error turns the viewer off for the rest
of the run instead of stopping the producer.

The viewer owns a second ``MjModel`` / ``MjData`` of the robot MJCF. The GUI may write option edits
back into the model it is given, so the IK model is not shared with it.

The MuJoCo passive viewer renders on its own thread of this process. On a host without ``DISPLAY`` /
``WAYLAND_DISPLAY`` the viewer is skipped. With a display that is set but not reachable MuJoCo ends the
process at launch ("could not initialize GLFW"), which is why the producer opens the viewer before any
DDS / ROS output exists. A crash of that thread later (GL driver, lost X connection) also ends the
producer; ``--no_viewer`` is the way out of both.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import mujoco as mj
import numpy as np

logger = logging.getLogger("igris_c_teleop.viewer")

DEFAULT_VIEWER_FPS = 30.0

# Human kinematic tree over the xrobot bodies that carry an IK task (gmr ik_configs/xrobot_to_igris_c.json).
TARGET_BONES: tuple[tuple[str, str], ...] = (
    ("Pelvis", "Spine3"),
    ("Spine3", "Head"),
    ("Spine3", "Left_Shoulder"),
    ("Left_Shoulder", "Left_Elbow"),
    ("Left_Elbow", "Left_Wrist"),
    ("Spine3", "Right_Shoulder"),
    ("Right_Shoulder", "Right_Elbow"),
    ("Right_Elbow", "Right_Wrist"),
    ("Pelvis", "Left_Hip"),
    ("Left_Hip", "Left_Knee"),
    ("Left_Knee", "Left_Foot"),
    ("Pelvis", "Right_Hip"),
    ("Right_Hip", "Right_Knee"),
    ("Right_Knee", "Right_Foot"),
)

_AXIS_LENGTH = 0.1  # m
_AXIS_WIDTH = 0.005
_BONE_WIDTH = 0.008
_AXIS_RGBA = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
_BONE_RGBA = np.array([0.1, 0.85, 1.0, 0.75], dtype=np.float32)
_ZERO3 = np.zeros(3)
_IDENTITY9 = np.eye(3).flatten()
_CAMERA_ELEVATION_DEG = -20.0
_JOIN_TIMEOUT_S = 2.0

Targets = Mapping[str, Sequence[Any]]  # human body name -> (pos_xyz, quat_wxyz), MuJoCo world frame


def ik_target_names(gmr: Any) -> tuple[str, ...]:
    """Human body names that carry an IK task in an enabled match table, first-seen order."""

    names: dict[str, None] = {}
    if gmr.use_ik_match_table1:
        names.update(dict.fromkeys(gmr.human_body_to_task1))
    if gmr.use_ik_match_table2:
        names.update(dict.fromkeys(gmr.human_body_to_task2))
    return tuple(names)


def _add_connector(scn: Any, geom_type: Any, width: float, start: np.ndarray, end: np.ndarray, rgba: np.ndarray) -> bool:
    if scn.ngeom >= scn.maxgeom:
        return False
    geom = scn.geoms[scn.ngeom]
    mj.mjv_initGeom(geom, geom_type, _ZERO3, _ZERO3, _IDENTITY9, rgba)
    mj.mjv_connector(geom, geom_type, width, start, end)
    scn.ngeom += 1
    return True


def draw_targets(
    scn: Any, targets: Targets, names: Sequence[str], bones: Sequence[tuple[str, str]] = TARGET_BONES
) -> int:
    """Append the IK target frames and the bones between them to ``scn``; returns the geoms added.

    Bodies that are missing or not finite are skipped, a degenerate quaternion keeps the bone but
    drops the frame, and drawing stops when the scene is full.
    """

    first = scn.ngeom
    positions: dict[str, np.ndarray] = {}
    matrix = np.empty(9)
    for name in names:
        entry = targets.get(name)
        if entry is None:
            continue
        pos = np.asarray(entry[0], dtype=np.float64).reshape(3)
        quat = np.asarray(entry[1], dtype=np.float64).reshape(4)
        if not np.all(np.isfinite(pos)):
            continue
        positions[name] = pos
        norm = float(np.linalg.norm(quat))
        if not np.isfinite(norm) or norm < 1e-9:
            continue
        mj.mju_quat2Mat(matrix, quat / norm)
        axes = matrix.reshape(3, 3)
        for axis in range(3):
            if not _add_connector(
                scn, mj.mjtGeom.mjGEOM_ARROW, _AXIS_WIDTH, pos, pos + _AXIS_LENGTH * axes[:, axis], _AXIS_RGBA[axis]
            ):
                return scn.ngeom - first
    for parent, child in bones:
        start, end = positions.get(parent), positions.get(child)
        if start is None or end is None:
            continue
        if not _add_connector(scn, mj.mjtGeom.mjGEOM_CAPSULE, _BONE_WIDTH, start, end, _BONE_RGBA):
            break
    return scn.ngeom - first


def _launch_passive(model: mj.MjModel, data: mj.MjData) -> Any:
    import mujoco.viewer  # lazy: a --no_viewer or headless run never loads the GLFW window code

    return mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)


class ReferenceViewer:
    """Passive MuJoCo window fed from the producer loop. See the module docstring for the constraints."""

    def __init__(
        self,
        xml_path: str,
        *,
        base_body: str,
        target_names: Sequence[str],
        fps: float = DEFAULT_VIEWER_FPS,
        cam_distance: float = 3.0,
        launch: Callable[[mj.MjModel, mj.MjData], Any] | None = None,
    ) -> None:
        if not fps > 0.0:
            raise ValueError("viewer fps must be positive")
        self._model = mj.MjModel.from_xml_path(str(xml_path))
        self._data = mj.MjData(self._model)
        mj.mj_forward(self._model, self._data)
        self._base_id = self._model.body(base_body).id
        self._target_names = tuple(target_names)
        self._period = 1.0 / float(fps)
        self._next_update = 0.0
        self.updates = 0
        self._window_count = 0
        self._window_seconds = 0.0
        self._window_max = 0.0

        before = set(threading.enumerate())
        self._handle = (launch or _launch_passive)(self._model, self._data)
        # The render thread(s) started by the launch; joined in close().
        self._threads = [thread for thread in threading.enumerate() if thread not in before]
        with self._handle.lock():
            # Semi-transparent robot so the targets inside the meshes stay visible.
            self._handle.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 1
            self._handle.cam.distance = float(cam_distance)
            self._handle.cam.elevation = _CAMERA_ELEVATION_DEG
            self._handle.cam.lookat[:] = self._data.xpos[self._base_id]
        self._handle.sync()

    @classmethod
    def open(cls, gmr: Any, robot: str, *, fps: float = DEFAULT_VIEWER_FPS) -> ReferenceViewer | None:
        """Viewer for a ``GeneralMotionRetargeting`` instance; ``None`` on a host without a display (headless, CI)."""

        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return None
        logger.info("Opening the viewer window. If the process ends here, the display is not reachable: pass --no_viewer.")
        from general_motion_retargeting import ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT

        return cls(
            gmr.xml_file,
            base_body=ROBOT_BASE_DICT[robot],
            target_names=ik_target_names(gmr),
            fps=fps,
            cam_distance=VIEWER_CAM_DISTANCE_DICT[robot],
        )

    @property
    def active(self) -> bool:
        return self._handle is not None

    def update(self, qpos: np.ndarray, targets: Targets | None = None, now: float | None = None) -> bool:
        """Show ``qpos`` and the IK targets when an update is due. False once the viewer is off.

        Never raises and never blocks on the window: a closed window or an error turns the viewer off.
        ``now`` is a monotonic time in seconds (tests); the default is ``time.monotonic()``.
        """

        handle = self._handle
        if handle is None:
            return False
        now = time.monotonic() if now is None else now
        if now < self._next_update:
            return True
        scheduled = self._next_update + self._period
        self._next_update = scheduled if scheduled > now else now + self._period

        start = time.perf_counter()
        try:
            if not handle.is_running():
                logger.warning("Viewer window closed; the producer keeps running without it.")
                self._disable()
                return False
            with handle.lock():
                self._data.qpos[:] = qpos
                mj.mj_forward(self._model, self._data)
                handle.cam.lookat[:] = self._data.xpos[self._base_id]  # follow the root, keep the operator's view angle
                scn = handle.user_scn
                if scn is not None:
                    scn.ngeom = 0
                    if targets is not None:
                        draw_targets(scn, targets, self._target_names)
            handle.sync()
        except Exception as exc:
            logger.warning("Viewer update failed (%s: %s); viewer off, the producer keeps running.", type(exc).__name__, exc)
            self._disable()
            return False
        elapsed = time.perf_counter() - start
        self.updates += 1
        self._window_count += 1
        self._window_seconds += elapsed
        self._window_max = max(self._window_max, elapsed)
        return True

    def take_stats(self) -> tuple[int, float, float]:
        """(updates, mean seconds, max seconds) since the previous call; resets the window."""

        count, total, worst = self._window_count, self._window_seconds, self._window_max
        self._window_count, self._window_seconds, self._window_max = 0, 0.0, 0.0
        return count, (total / count if count else 0.0), worst

    def _disable(self) -> None:
        """Request the window to close without waiting for it (safe inside the loop)."""

        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            handle.close()
        except Exception as exc:
            logger.warning("Viewer close failed: %s", exc)

    def close(self) -> None:
        """Teardown: close the window and wait for the render thread (up to ``_JOIN_TIMEOUT_S`` per thread).

        MuJoCo registers ``glfw.terminate`` with ``atexit``; the interpreter segfaults in it when the
        render loop is still running, so the thread has to be gone before the process exits.
        """

        self._disable()
        threads, self._threads = self._threads, []
        for thread in threads:
            thread.join(timeout=_JOIN_TIMEOUT_S)
        if any(thread.is_alive() for thread in threads):
            logger.warning("Viewer render thread did not stop within %.1f s.", _JOIN_TIMEOUT_S)


__all__ = ["DEFAULT_VIEWER_FPS", "TARGET_BONES", "ReferenceViewer", "draw_targets", "ik_target_names"]
