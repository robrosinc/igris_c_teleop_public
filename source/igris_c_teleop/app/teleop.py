"""One-command teleoperation session: the VRL2 downlink sender next to the producer.

Starts two child processes of this pixi env with one DDS namespace/domain for both:

  1. ``vr-link-stream`` (``vr_link.cli.stream``): camera + calibration + frame-matched
     robot state to the headset (ports 9100/9101). The target picks the packaged
     vr_link config: robot -> ``stream_dds.yaml`` (igris_c_camera DDS topic, measured
     calibration), usb -> ``stream_usb.yaml`` (stereo camera on this PC), mujoco ->
     ``stream_dds_mujoco.yaml``, isaac -> ``stream_dds_isaac.yaml``, shm ->
     ``stream_shm.yaml`` (simulator shared-memory segment).
  2. the producer (``igris_c_teleop.app.producer``): VRS1 uplink (port 9102) -> GMR -> motion
     reference + HandCmd / control-mode DDS. ``--target`` is forwarded to it and selects the
     reference transport (app/targets.py): CycloneDDS for isaac, ROS 2 for every other target.
     Every argument this launcher does not know is passed to it unchanged (``--measure_fps``,
     ``--finger_source``, ...).

Separate processes on purpose: the video encoder cannot stall the 100 Hz retargeting
loop and a crash on one side cannot take the other down silently. The launcher
forwards SIGINT/SIGTERM to both children and exits when either child exits, stopping
the other one first (SIGINT, then SIGTERM after ``--grace`` seconds, then SIGKILL).

The robot-command semantics are unchanged: the producer's DDS outputs and the ROBROS
hardware approval and verification procedure in README.md apply exactly as when it is
started alone with ``pixi run producer``.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Mapping, Sequence

from ..dds import TOPIC_NAMINGS, cyclonedds_xml
from .targets import DEFAULT_TARGET, TARGETS

logger = logging.getLogger("igris_c_teleop.teleop")

# target -> (vr-link-stream --source, packaged vr_link config file); one entry per app/targets.py TARGETS
STREAM_TARGETS: dict[str, tuple[str, str]] = {
    "robot": ("dds", "stream_dds.yaml"),
    "usb": ("usb", "stream_usb.yaml"),  # real robot, stereo camera on this PC's USB
    "mujoco": ("dds", "stream_dds_mujoco.yaml"),
    "isaac": ("dds", "stream_dds_isaac.yaml"),
    "shm": ("shm", "stream_shm.yaml"),
}
# Same-PC simulator defaults (loopback DDS interface, no local viewer, NVENC encoder) apply to these
# targets only; every other target runs exactly as without them.
SIM_DEFAULTS_TARGETS = ("isaac",)
SIM_DEFAULT_ENCODER = "h264_nvenc"
DEFAULT_GRACE_SEC = 5.0
_POLL_SEC = 0.2


@dataclass(frozen=True)
class ChildCommand:
    name: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)


def packaged_stream_config(filename: str) -> Path:
    """Path of a config shipped in the installed ``vr_link`` package (editable or wheel)."""

    return Path(str(resources.files("vr_link") / "configs" / filename))


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        prog="igris-c-teleop",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Unknown arguments are forwarded to the producer (python -m igris_c_teleop --help).",
    )
    dds = parser.add_argument_group("DDS (shared by both processes)")
    dds.add_argument(
        "--dds_namespace", type=str, required=True, help="Robot-unit namespace (igris_c configs/robot/*.yaml)."
    )
    dds.add_argument("--dds_domain_id", type=int, default=0)
    dds.add_argument(
        "--dds_interface",
        type=str,
        default=None,
        help="Pin CycloneDDS to one network interface for both processes (defaults to 'lo' on --target isaac when $CYCLONEDDS_URI is unset).",
    )
    dds.add_argument(
        "--dds_topic_naming",
        choices=TOPIC_NAMINGS,
        default="native",
        help="Forwarded to the producer. vr-link-stream reads it from its config file only, so a non-native value "
        "needs --stream_config with a matching dds.topic_naming.",
    )

    stream = parser.add_argument_group("vr-link-stream (VRL2 downlink)")
    stream.add_argument(
        "--target",
        choices=TARGETS,
        default=DEFAULT_TARGET,
        help="What the session drives: picks the frame source that feeds the headset and, forwarded to the "
        "producer, the motion-reference transport (CycloneDDS for isaac, ROS 2 for every other target).",
    )
    stream.add_argument(
        "--stream_config",
        type=Path,
        default=None,
        help="vr-link-stream --config override (default: the packaged file of --target).",
    )
    stream.add_argument(
        "--stream_encoder", type=str, default=None, help="vr-link-stream --encoder override (libx264, h264_nvenc, ...)."
    )
    stream.add_argument("--stream_max_fps", type=float, default=None, help="vr-link-stream --max-fps.")
    stream.add_argument(
        "--head_rgb", choices=("on", "off"), default=None,
        help="vr-link-stream --head-rgb: the second video stream (head colour camera, VideoFrame.stream_id = "
        "HeadRgb) on top of the config file's head_rgb.enable (packaged configs: off). Needs a headset build "
        "that demultiplexes stream_id.",
    )
    stream.add_argument("--no_stream", action="store_true", help="Producer only (same as pixi run producer).")
    stream.add_argument(
        "--viewer",
        choices=("auto", "on", "off"),
        default="auto",
        help="Producer MuJoCo viewer: 'auto' (default) disables the local viewer on --target isaac.",
    )

    parser.add_argument(
        "--grace",
        type=float,
        default=DEFAULT_GRACE_SEC,
        help="Seconds between SIGINT and SIGTERM when stopping a child.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print the two command lines and exit.")
    parser.add_argument("--log_level", type=str, default="INFO")
    args, producer_args = parser.parse_known_args(argv)
    return args, list(producer_args)


def build_commands(
    args: argparse.Namespace,
    producer_args: Sequence[str],
    *,
    environ: Mapping[str, str] | None = None,
) -> list[ChildCommand]:
    """The child command lines, stream first. Pure (no process is started)."""

    env_map = os.environ if environ is None else environ
    sim_defaults = args.target in SIM_DEFAULTS_TARGETS
    interface = args.dds_interface
    if interface is None and sim_defaults and not env_map.get("CYCLONEDDS_URI"):
        interface = "lo"
    try:
        child_env = {"CYCLONEDDS_URI": cyclonedds_xml(interface)} if interface else {}
    except ValueError as exc:
        raise SystemExit(f"--dds_interface: {exc}") from exc

    python = sys.executable
    commands: list[ChildCommand] = []
    if not args.no_stream:
        source, config_name = STREAM_TARGETS[args.target]
        if args.stream_config is not None:
            config = Path(args.stream_config)
        elif args.dds_topic_naming != "native":
            raise SystemExit(
                f"--dds_topic_naming {args.dds_topic_naming} needs --stream_config: vr-link-stream takes topic_naming "
                "from its config file (dds.topic_naming) and has no CLI override."
            )
        else:
            config = packaged_stream_config(config_name)
        stream_argv = [
            python, "-m", "vr_link.cli.stream",
            "--source", source,
            "--config", str(config),
            "--dds-namespace", args.dds_namespace,
            "--dds-domain", str(args.dds_domain_id),
        ]
        encoder = args.stream_encoder or (SIM_DEFAULT_ENCODER if sim_defaults else None)
        if encoder:
            stream_argv += ["--encoder", encoder]
        if args.stream_max_fps:
            stream_argv += ["--max-fps", str(args.stream_max_fps)]
        if args.head_rgb:
            stream_argv += ["--head-rgb", args.head_rgb]
        commands.append(ChildCommand("vr-link-stream", stream_argv, dict(child_env)))

    # The producer derives the motion-reference transport from the same target (app/targets.py).
    producer_argv = [
        python, "-m", "igris_c_teleop",
        "--target", args.target,
        "--dds_namespace", args.dds_namespace,
        "--dds_domain_id", str(args.dds_domain_id),
        "--dds_topic_naming", args.dds_topic_naming,
    ]
    if interface and "--dds_interface" not in producer_args:
        producer_argv += ["--dds_interface", interface]
    disable_viewer = args.viewer == "off" or (args.viewer == "auto" and sim_defaults)
    if disable_viewer and "--no_viewer" not in producer_args:
        producer_argv.append("--no_viewer")
    producer_argv.extend(producer_args)
    commands.append(ChildCommand("producer", producer_argv, dict(child_env)))
    return commands


def _signal_child(proc: subprocess.Popen, sig: signal.Signals) -> None:
    if proc.poll() is None:
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass


def _stop_children(children: dict[str, subprocess.Popen], grace: float) -> None:
    """SIGINT every live child, SIGTERM the survivors after ``grace`` s, SIGKILL after another ``grace`` s."""

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        for proc in children.values():
            _signal_child(proc, sig)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and any(p.poll() is None for p in children.values()):
            time.sleep(_POLL_SEC)
        if all(p.poll() is not None for p in children.values()):
            return


def run_supervised(commands: Sequence[ChildCommand], *, grace: float = DEFAULT_GRACE_SEC) -> int:
    """Run the children until one exits or a stop signal arrives; return the exit code.

    The code is the first exited child's code, or 130 after an operator SIGINT/SIGTERM
    while every child was still running. Children inherit stdin/stdout/stderr, so both
    logs interleave on the launcher's terminal with their own logger prefixes.
    """

    if not commands:
        raise ValueError("no commands to supervise")
    stop_requested = False

    def _request_stop(signum, frame):  # noqa: ARG001
        nonlocal stop_requested
        stop_requested = True

    previous = {sig: signal.signal(sig, _request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    children: dict[str, subprocess.Popen] = {}
    exit_code = 0
    try:
        for command in commands:
            logger.info("starting %s: %s", command.name, " ".join(command.argv))
            child_env = {**os.environ, **command.env}
            children[command.name] = subprocess.Popen(command.argv, env=child_env)
        while True:
            exited = [(name, proc.returncode) for name, proc in children.items() if proc.poll() is not None]
            if exited:
                name, code = exited[0]
                exit_code = int(code)
                logger.warning("%s exited with code %d; stopping the other process", name, code)
                break
            if stop_requested:
                exit_code = 130
                logger.info("stop requested; stopping both processes")
                break
            time.sleep(_POLL_SEC)
    finally:
        _stop_children(children, grace)
        for name, proc in children.items():
            logger.info("%s finished with code %s", name, proc.returncode)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    args, producer_args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    commands = build_commands(args, producer_args)
    if args.dry_run:
        for command in commands:
            print(f"{command.name}: {' '.join(command.argv)}")
        return 0
    return run_supervised(commands, grace=args.grace)


__all__ = [
    "ChildCommand",
    "DEFAULT_TARGET",
    "STREAM_TARGETS",
    "build_commands",
    "main",
    "packaged_stream_config",
    "parse_args",
    "run_supervised",
]
