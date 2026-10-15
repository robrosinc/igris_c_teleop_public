"""Motion-reference frame builder for the ``/gmr/teleop/retarget_frame`` ROS 2 topic.

Ported from hit-bench ``scripts/teleoperation/_gmr_redis_frame.py`` (itself a
port of GMR's PICO producer payload logic). The Redis side channels (TWIST2
``action_body`` mimic observation, ``extra`` keys) are gone: the frame carries
the whole-body reference only.

Consumer contract (igris_c_inference_public ``RosMotionSource``, a string
scanner over a ``std_msgs/String`` payload): every array carries a leading
frame axis of length 1. Required keys are ``dof_pos`` and ``dof_vel`` (MJCF
declaration order, 31 for igris_c_v2), ``root_rot`` (wxyz),
``link_body_list`` and ``keybody_rot_local`` (root frame,
``len(link_body_list) * 4`` numbers); optional keys are ``keybody_pos_local``,
``root_pos``, ``root_vel`` and ``root_angvel``. Unknown keys are ignored, so
the payload is whitelisted to ``CONTRACT_KEYS`` and serialised as compact JSON
with ``allow_nan=False`` (a NaN would otherwise be skipped by the scanner and
silently shorten ``dof_pos``).

Only numpy, mujoco and ``general_motion_retargeting`` are required.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

# Body names the whole-body policy observation requires in ``link_body_list``
# (``base_link`` is inserted by the receiver itself). Kept explicit so a GMR
# keybody config change fails here instead of silently starving the policy.
GMT_REQUIRED_BODY_NAMES: tuple[str, ...] = (
    "Left_Hand",
    "Right_Hand",
    "Link_Ankle_Roll_Left",
    "Link_Ankle_Roll_Right",
    "Link_Elbow_Pitch_Left",
    "Link_Elbow_Pitch_Right",
    "Link_Shoulder_Pitch_Left",
    "Link_Shoulder_Pitch_Right",
    "Link_Knee_Pitch_Left",
    "Link_Knee_Pitch_Right",
    "Link_Hip_Pitch_Left",
    "Link_Hip_Pitch_Right",
)

# Keys published on the ROS topic, in payload order. Everything else that
# gmr's ``build_motion_data`` returns (world-frame keybodies, ``keybody_pos``,
# ``local_body_*``) is dropped.
CONTRACT_KEYS: tuple[str, ...] = (
    "fps",
    "root_pos",
    "root_rot",
    "root_vel",
    "root_angvel",
    "dof_pos",
    "dof_vel",
    "keybody_pos_local",
    "keybody_rot_local",
    "link_body_list",
    "record_meta",
)
REQUIRED_CONSUMER_KEYS: tuple[str, ...] = ("dof_pos", "dof_vel", "root_rot", "link_body_list", "keybody_rot_local")

FREE_JOINT_QPOS_SIZE = 7  # root pos (3) + root quat wxyz (4)
DEFAULT_SOURCE_TAG = "igris_c_teleop"


def jsonable(value: Any) -> Any:
    """Recursively convert numpy containers to plain Python for ``json.dumps``."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def to_ros_payload(frame: dict[str, Any]) -> str:
    """Serialise a built frame for ``std_msgs/String``.

    Raises ``ValueError`` on non-finite numbers instead of emitting ``NaN``
    tokens the C++ scanner would skip.
    """

    missing = [key for key in REQUIRED_CONSUMER_KEYS if key not in frame]
    if missing:
        raise ValueError(f"frame lacks consumer-required keys: {missing}.")
    return json.dumps(frame, separators=(",", ":"), allow_nan=False)


class RetargetFramePayloadBuilder:
    """Builds one contract frame dict per retargeted MuJoCo qpos.

    Velocities are wall-clock finite differences against the previously built
    frame, zero for the first ``zero_velocity_frames`` frames after ``reset``,
    then optionally smoothed with a first-order low-pass filter (same
    ``alpha = dt / (tau + dt)`` as GMR).
    """

    def __init__(
        self,
        model,
        *,
        robot_name: str,
        target_fps: float,
        velocity_lpf_hz: float | None = None,
        zero_velocity_frames: int = 10,
        source_tag: str = DEFAULT_SOURCE_TAG,
    ) -> None:
        import mujoco as mj
        from general_motion_retargeting.utils.motion_utils import get_default_keybody_names

        if target_fps <= 0.0:
            raise ValueError(f"target_fps must be positive, got {target_fps}.")
        if velocity_lpf_hz is not None and velocity_lpf_hz <= 0.0:
            raise ValueError(f"velocity_lpf_hz must be positive or None, got {velocity_lpf_hz}.")
        if zero_velocity_frames < 1:
            raise ValueError(f"zero_velocity_frames must be >= 1, got {zero_velocity_frames}.")

        self._mj = mj
        self._model = model
        self._data = mj.MjData(model)
        self._robot_name = str(robot_name)
        self._target_fps = float(target_fps)
        self._velocity_lpf_hz = velocity_lpf_hz
        self._zero_velocity_frames = int(zero_velocity_frames)
        self._source_tag = str(source_tag)

        body_ids: list[int] = []
        body_names: list[str] = []
        for name in get_default_keybody_names(self._robot_name):
            body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                body_ids.append(int(body_id))
                body_names.append(name)
        missing = [name for name in GMT_REQUIRED_BODY_NAMES if name not in body_names]
        if missing:
            raise ValueError(
                f"GMR keybody list for {self._robot_name!r} lacks bodies the policy runtime requires: {missing}."
            )
        self._body_ids = np.array(body_ids, dtype=np.int64)
        self._body_names = tuple(body_names)

        self.nq = int(model.nq)
        self.dof_count = self.nq - FREE_JOINT_QPOS_SIZE
        self.frame_idx = 0
        self._prev_qpos: np.ndarray | None = None
        self._prev_time: float | None = None
        self._frames_since_reset = 0
        self._lpf_state: dict[str, np.ndarray] = {}

    @property
    def body_names(self) -> tuple[str, ...]:
        return self._body_names

    def reset(self) -> None:
        """Drop finite-difference and filter history (VR (re)connection)."""

        self._prev_qpos = None
        self._prev_time = None
        self._frames_since_reset = 0
        self._lpf_state = {}

    def build(self, qpos: np.ndarray, now: float | None = None) -> dict[str, Any]:
        """Return the JSON-able contract frame for ``qpos`` (``model.nq`` values)."""

        from general_motion_retargeting.rot_utils import quat_diff_np
        from general_motion_retargeting.utils.motion_utils import build_motion_data

        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (self.nq,):
            raise ValueError(f"Expected qpos of shape ({self.nq},), got {qpos.shape}.")
        if not np.all(np.isfinite(qpos)):
            raise ValueError("qpos contains non-finite values.")
        now = time.time() if now is None else float(now)

        self._data.qpos[:] = qpos
        self._mj.mj_forward(self._model, self._data)
        keybody_pos_world = self._data.xpos[self._body_ids][None, :, :].copy()
        keybody_rot_world_wxyz = self._data.xquat[self._body_ids][None, :, :].copy()

        motion_data = build_motion_data(
            aligned_fps=self._target_fps,
            root_pos=qpos[:3][None, :],
            root_rot_wxyz=qpos[3:7][None, :],
            dof_pos=qpos[FREE_JOINT_QPOS_SIZE:][None, :],
            keybody_pos_world=keybody_pos_world,
            keybody_rot_world_wxyz=keybody_rot_world_wxyz,
            keybody_names=list(self._body_names),
            local_body_pos=None,
            local_body_link_body_list=None,
        )

        root_vel = np.zeros((1, 3), dtype=np.float64)
        root_angvel = np.zeros((1, 3), dtype=np.float64)
        dof_vel = np.zeros((1, self.dof_count), dtype=np.float64)
        zero_velocity = self._frames_since_reset < self._zero_velocity_frames
        dt: float | None = None
        if not zero_velocity and self._prev_qpos is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 1e-6:
                prev = self._prev_qpos
                root_vel[0] = (qpos[:3] - prev[:3]) / dt
                root_angvel[0] = quat_diff_np(prev[3:7], qpos[3:7], scalar_first=True) / dt
                dof_vel[0] = (qpos[FREE_JOINT_QPOS_SIZE:] - prev[FREE_JOINT_QPOS_SIZE:]) / dt
            else:
                dt = None
        root_vel, root_angvel, dof_vel = self._apply_velocity_lpf(root_vel, root_angvel, dof_vel, dt, zero_velocity)
        motion_data["root_vel"] = root_vel
        motion_data["root_angvel"] = root_angvel
        motion_data["dof_vel"] = dof_vel

        self._prev_qpos = qpos.copy()
        self._prev_time = now
        self._frames_since_reset += 1

        motion_data["record_meta"] = {
            "source": self._source_tag,
            "robot": self._robot_name,
            "frame_idx": int(self.frame_idx),
            "unix_ms": int(time.time() * 1000),
            "fps": self._target_fps,
        }
        self.frame_idx += 1
        return jsonable({key: motion_data[key] for key in CONTRACT_KEYS})

    # -- internals -------------------------------------------------------------

    def _apply_velocity_lpf(
        self,
        root_vel: np.ndarray,
        root_angvel: np.ndarray,
        dof_vel: np.ndarray,
        dt: float | None,
        zero_velocity: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._velocity_lpf_hz is None:
            return root_vel, root_angvel, dof_vel
        if zero_velocity:
            # Re-seed the filter with the (zero) sample, as GMR does.
            self._lpf_state = {
                "root_vel": root_vel[0].copy(),
                "root_angvel": root_angvel[0].copy(),
                "dof_vel": dof_vel[0].copy(),
            }
            return root_vel, root_angvel, dof_vel

        dt = max(1e-6, float(dt) if dt is not None else 1.0 / self._target_fps)
        tau = 1.0 / (2.0 * np.pi * self._velocity_lpf_hz)
        alpha = dt / (tau + dt)
        root_vel[0] = self._filter("root_vel", root_vel[0], alpha)
        root_angvel[0] = self._filter("root_angvel", root_angvel[0], alpha)
        dof_vel[0] = self._filter("dof_vel", dof_vel[0], alpha)
        return root_vel, root_angvel, dof_vel

    def _filter(self, key: str, sample: np.ndarray, alpha: float) -> np.ndarray:
        prev = self._lpf_state.get(key)
        filtered = sample.copy() if prev is None or prev.shape != sample.shape else prev + alpha * (sample - prev)
        self._lpf_state[key] = filtered.copy()
        return filtered


__all__ = [
    "CONTRACT_KEYS",
    "DEFAULT_SOURCE_TAG",
    "FREE_JOINT_QPOS_SIZE",
    "GMT_REQUIRED_BODY_NAMES",
    "REQUIRED_CONSUMER_KEYS",
    "RetargetFramePayloadBuilder",
    "jsonable",
    "to_ros_payload",
]
