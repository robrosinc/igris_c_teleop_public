"""Producer entry point: CLI contract, safety guards, dependency hygiene."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from igris_c_teleop.app import producer

PACKAGE_DIR = Path(producer.__file__).resolve().parents[1]
# Nothing from Redis, Isaac, or hit-bench may come back; torch is gmr's business, not ours.
FORBIDDEN_IMPORT_ROOTS = ("redis", "isaaclab", "isaacsim", "omni", "source", "hit_bench", "_gmr_redis_frame", "torch")


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_package_imports_stay_clean():
    for path in PACKAGE_DIR.rglob("*.py"):
        offending = _import_roots(path).intersection(FORBIDDEN_IMPORT_ROOTS)
        assert not offending, f"{path.name} imports {sorted(offending)}"
    assert {"vr_link", "general_motion_retargeting"} <= _import_roots(PACKAGE_DIR / "uplink" / "vrs1_source.py")


def test_defaults_match_port_plan():
    args = producer.parse_args([])
    assert args.pose_port == 9102 and args.stale_timeout == 0.5
    assert args.finger_source == "auto" and args.target_fps == 100.0
    assert args.ros_domain_id == 91 and args.ros_topic == "/gmr/teleop/retarget_frame"
    assert not args.no_dds and args.dds_domain_id == 0 and args.dds_namespace is None
    assert args.dds_topic_naming == "native" and args.hand_dds_topic == "handcmd"
    assert args.control_mode_timeout_ms == 3000
    assert not args.no_mode_events and args.replay_vrs1 is None
    # Headset controller relay for recorders: on by default, name from the vr_link contract.
    assert args.controller_dds_topic == "vr/controller" and not args.no_controller_dds
    assert producer.parse_args(["--no_controller_dds"]).no_controller_dds
    for removed in ("redis_ip", "redis_port", "hand_dds", "auto_start_event", "stream_port"):
        assert not hasattr(args, removed), removed


def test_controller_relay_uses_the_vr_link_contract():
    from vr_link.messages import teleop_dds

    assert producer.DEFAULT_CONTROLLER_TOPIC == teleop_dds.CONTROLLER_TOPIC
    assert producer.ControllerStatePublisher.__module__ == "vr_link.network.dds.teleop"
    assert producer.controller_sample_from_vr_state is teleop_dds.controller_sample_from_vr_state


def test_dds_namespace_is_required_unless_no_dds():
    with pytest.raises(SystemExit, match="dds_namespace"):
        producer.main([])
    with pytest.raises(SystemExit, match="finger source"):
        producer.main(["--dds_namespace", "igris_c_IG01", "--finger_source", "none"])
    with pytest.raises(SystemExit, match="target_fps"):
        producer.main(["--no_dds", "--target_fps", "0"])


def test_reference_shaping_defaults():
    from vr_link.messages import teleop_dds

    args = producer.parse_args([])
    # Convergence ramp: on by default, 2 s smoothstep from LowState, 0.5 rad/s cap, hand 1.0/s.
    assert not args.no_converge and args.converge_seconds == 2.0 and args.converge_max_rad_s == 0.5
    assert args.converge_hand_rate == 1.0 and args.lowstate_max_age == 0.5
    assert args.lowstate_dds_topic == "lowstate" and args.robotstate_dds_topic == "robotstate"
    # Arm hold on bare-hand tracking loss: on by default, 0.2 s re-acquire, 0.5 rad/s release.
    assert args.arm_hold == "on" and args.arm_hold_release_s == 0.2 and args.arm_hold_release_rad_s == 0.5
    assert producer.parse_args(["--arm_hold", "off"]).arm_hold == "off"
    # HUD feedback topic comes from the vr_link contract.
    assert args.motion_feedback_dds_topic == teleop_dds.MOTION_FEEDBACK_TOPIC == "teleop/motion_feedback"
    assert not args.no_motion_feedback_dds
    assert producer.MotionFeedbackPublisher.__module__ == "vr_link.network.dds.teleop"
