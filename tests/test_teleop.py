"""teleop session entry point: command construction (pure) and child supervision (real subprocesses, no DDS/ROS)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from igris_c_gmr import teleop

NS = ["--dds_namespace", "igris_c_test"]


def _commands(*argv: str):
    args, rest = teleop.parse_args(list(argv))
    return teleop.build_commands(args, rest)


def test_default_target_is_robot_dds_with_packaged_config():
    stream, producer = _commands(*NS)
    assert stream.name == "vr-link-stream" and producer.name == "producer"
    assert stream.argv[:3] == [sys.executable, "-m", "vr_link.cli.stream"]
    assert stream.argv[stream.argv.index("--source") + 1] == "dds"
    config = Path(stream.argv[stream.argv.index("--config") + 1])
    assert config.name == "stream_dds.yaml" and config.is_file()
    assert stream.argv[stream.argv.index("--dds-namespace") + 1] == "igris_c_test"
    assert stream.argv[stream.argv.index("--dds-domain") + 1] == "0"
    assert producer.argv[:3] == [sys.executable, "-m", "igris_c_gmr"]
    assert producer.argv[3:] == [
        "--dds_namespace", "igris_c_test", "--dds_domain_id", "0", "--dds_topic_naming", "native",
    ]


@pytest.mark.parametrize(
    ("target", "source", "config_name"),
    [
        ("robot", "dds", "stream_dds.yaml"),
        ("usb", "usb", "stream_usb.yaml"),
        ("mujoco", "dds", "stream_dds_mujoco.yaml"),
        ("isaac", "dds", "stream_dds_isaac.yaml"),
        ("shm", "shm", "stream_shm.yaml"),
    ],
)
def test_every_target_maps_to_a_packaged_config(target, source, config_name):
    stream, _ = _commands(*NS, "--target", target)
    assert stream.argv[stream.argv.index("--source") + 1] == source
    config = Path(stream.argv[stream.argv.index("--config") + 1])
    assert config.name == config_name and config.is_file()


def test_shared_dds_settings_and_producer_passthrough():
    stream, producer = _commands(
        *NS, "--dds_domain_id", "7", "--stream_encoder", "h264_nvenc", "--stream_max_fps", "25",
        "--measure_fps", "--finger_source", "grip",
    )
    assert stream.argv[stream.argv.index("--dds-domain") + 1] == "7"
    assert stream.argv[stream.argv.index("--encoder") + 1] == "h264_nvenc"
    assert stream.argv[stream.argv.index("--max-fps") + 1] == "25.0"
    assert producer.argv[producer.argv.index("--dds_domain_id") + 1] == "7"
    assert producer.argv[-3:] == ["--measure_fps", "--finger_source", "grip"]
    assert "--measure_fps" not in stream.argv and "--finger_source" not in stream.argv


def test_no_stream_gives_producer_only():
    (producer,) = _commands(*NS, "--no_stream")
    assert producer.name == "producer"


def test_non_native_naming_requires_explicit_stream_config(tmp_path):
    with pytest.raises(SystemExit):
        _commands(*NS, "--dds_topic_naming", "ros_compatible")
    custom = tmp_path / "stream.yaml"
    custom.write_text("tcp: {}\n")
    stream, producer = _commands(*NS, "--dds_topic_naming", "ros_compatible", "--stream_config", str(custom))
    assert Path(stream.argv[stream.argv.index("--config") + 1]) == custom
    assert producer.argv[producer.argv.index("--dds_topic_naming") + 1] == "ros_compatible"


def test_dry_run_prints_without_starting(capsys):
    assert teleop.main([*NS, "--dry_run"]) == 0
    out = capsys.readouterr().out
    assert "vr-link-stream:" in out and "producer:" in out


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


SLEEPER = "import signal, sys, time; signal.signal(signal.SIGINT, lambda *a: sys.exit(0)); time.sleep(30)"


def test_supervisor_stops_the_survivor_when_one_child_exits():
    commands = [
        teleop.ChildCommand("sleeper", _py(SLEEPER)),
        teleop.ChildCommand("quitter", _py("import sys; sys.exit(3)")),
    ]
    started = time.monotonic()
    assert teleop.run_supervised(commands, grace=2.0) == 3
    assert time.monotonic() - started < 10.0  # the sleeper was stopped by SIGINT, not left for 30 s


def test_supervisor_escalates_to_sigterm_for_a_child_ignoring_sigint():
    stubborn = "import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(30)"
    commands = [
        teleop.ChildCommand("stubborn", _py(stubborn)),
        teleop.ChildCommand("quitter", _py("import sys; sys.exit(1)")),
    ]
    started = time.monotonic()
    assert teleop.run_supervised(commands, grace=0.5) == 1
    assert time.monotonic() - started < 10.0
