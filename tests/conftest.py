"""Shared fixtures: the GMR robot model and a keyframe qpos.

Every test module skips when mujoco, gmr or vr_link are missing; ROS
integration tests carry the ``ros`` marker and skip without rclpy.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("general_motion_retargeting")
pytest.importorskip("vr_link")

import mujoco as mj  # noqa: E402
from general_motion_retargeting import ROBOT_XML_DICT  # noqa: E402

ROBOT = "robros_igris_c_v2"


@pytest.fixture(scope="session")
def model():
    return mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[ROBOT]))


@pytest.fixture(scope="session")
def keyframe_qpos(model):
    assert model.nkey >= 1
    return np.array(model.key_qpos[0], dtype=np.float64)
