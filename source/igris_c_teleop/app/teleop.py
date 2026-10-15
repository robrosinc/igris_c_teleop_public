"""One-command teleoperation session: the VRL2 downlink sender next to the producer.

Starts two child processes of this pixi env with one DDS namespace/domain for both:

  1. ``vr-link-stream`` (``vr_link.cli.stream``): camera + calibration + frame-matched
     robot state to the headset (ports 9100/9101). The target picks the packaged
     vr_link config: robot -> ``stream_dds.yaml`` (igris_c_camera DDS topic, measured
     calibration), usb -> ``stream_usb.yaml`` (stereo camera on this PC), mujoco ->
     ``stream_dds_mujoco.yaml``, isaac -> ``stream_dds_isaac.yaml``, shm ->
     ``stream_shm.yaml`` (simulator shared-memory segment).
  2. the producer (``igris_c_teleop.app.producer``): VRS1 uplink (port 9102) -> GMR -> ROS 2
     frame + HandCmd / control-mode DDS. Every argument this launcher does not know is
     passed to it unchanged (``--measure_fps``, ``--finger_source``, ...).

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
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Sequence

from ..dds import TOPIC_NAMINGS

logger = logging.getLogger("igris_c_teleop.teleop")

# target -> (vr-link-stream --source, packaged vr_link config file)
STREAM_TARGETS: dict[str, tuple[str, str]] = {
    "robot": ("dds", "stream_dds.yaml"),
    "usb": ("usb", "stream_usb.yaml"),  # real robot, stereo camera on this PC's USB
    "mujoco": ("dds", "stream_dds_mujoco.yaml"),
    "isaac": ("dds", "stream_dds_isaac.yaml"),
    "shm": ("shm", "stream_shm.yaml"),
}
DEFAULT_TARGET = "robot"
DEFAULT_GRACE_SEC = 5.0
_POLL_SEC = 0.2


@dataclass(frozen=True)
class ChildCommand:
    name: str
    argv: list[str]


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
        "--dds_topic_naming",
        choices=TOPIC_NAMINGS,
        default="native",
        help="Forwarded to the producer. vr-link-stream reads it from its config file only, so a non-native value "
        "needs --stream_config with a matching dds.topic_naming.",
    )

    stream = parser.add_argument_group("vr-link-stream (VRL2 downlink)")
    stream.add_argument(
        "--target", choices=tuple(STREAM_TARGETS), default=DEFAULT_TARGET, help="Which frame source feeds the headset."
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
    stream.add_argument("--no_stream", action="store_true", help="Producer only (same as pixi run producer).")

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


def build_commands(args: argparse.Namespace, producer_args: Sequence[str]) -> list[ChildCommand]:
    """The child command lines, stream first. Pure (no process is started)."""

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
        if args.stream_encoder:
            stream_argv += ["--encoder", args.stream_encoder]
        if args.stream_max_fps:
            stream_argv += ["--max-fps", str(args.stream_max_fps)]
        commands.append(ChildCommand("vr-link-stream", stream_argv))

    producer_argv = [
        python, "-m", "igris_c_teleop",
        "--dds_namespace", args.dds_namespace,
        "--dds_domain_id", str(args.dds_domain_id),
        "--dds_topic_naming", args.dds_topic_naming,
        *producer_args,
    ]
    commands.append(ChildCommand("producer", producer_argv))
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
            children[command.name] = subprocess.Popen(command.argv, env=os.environ.copy())
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
