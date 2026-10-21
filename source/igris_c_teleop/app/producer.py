"""vr_system headset teleoperation producer: VRS1 -> GMR -> ROS 2 frame + igris_c_sdk DDS.

Pipeline::

    headset (vr_system pawn, VRS1 FlatBuffers over TCP, this host :9102)
      -> vr_link.TcpVRDevice (gmr joint convention adapter)
      -> GeneralMotionRetargeting(xrobot -> robros_igris_c_v2)      body IK, 38 qpos
      -> NeckOverride                                               HMD-relative neck into qpos
      -> ArmHold                                                    arm joints frozen while the bare hand is untracked
      -> ConvergeGate (LowState + robotstate + VRS1 teleop_state)   ramp from the robot's measured pose on Start /
                                                                    Resume / low-level entry, 31 hinges in qpos
      -> RetargetFramePayloadBuilder -> RetargetFrameRosPublisher   /gmr/teleop/retarget_frame (String, domain 91)
                                                                    -> igris_c_inference_node (ONNX WBC -> DDS lowcmd)
      -> HandTargets -> HandCmdDdsPublisher                         <ns>/handcmd (DDS HandCmd, fingers bypass the policy)
      -> ControllerEvents -> ControlModeClient                      <ns>/service/control_mode/request (DDS service)
      -> RefStateDdsPublisher (opt-in, --ref_dds_topic)             <ns>/gmr/refstate (DDS MasterArmState: the 31
                                                                    retargeted joints, headset hand-overlay display only)
      -> ControllerStatePublisher (vr_link, --no_controller_dds)    <ns>/vr/controller (DDS Joy_: headset-UI recording
                                                                    toggle counter, session id, controller axes; read by
                                                                    episode recorders such as the hit-tasks Isaac bridge)
      -> MotionFeedbackPublisher (vr_link, --no_motion_feedback_dds) <ns>/teleop/motion_feedback (DDS Joy_: converging
                                                                    flag, per-arm hold; vr-link-stream puts it into
                                                                    Signals for the headset HUD. The progress axis is
                                                                    deprecated and sent as 0)

Run order on the robot NUC (one terminal each):
  1. igris_c controller / bridge (DDS lowstate, lowcmd, handcmd, control-mode service) - human-operated.
  2. this producer:  python -m igris_c_teleop --dds_namespace <robot namespace> [--measure_fps]
     then put the headset in Active so the skeleton streams ("VR skeleton connected" in the log).
  3. ros2 run igris_c_inference_public igris_c_inference_node --config <policy>/params.yaml
     The node needs 46 frames within 5 s of start, so step 2 must be streaming first.
     Nothing is published while the skeleton is absent; the inference node holds its last reference.

Controller buttons (rising edge, once per press, while the skeleton is connected):
  right B -> low-level joint control (redundant: the inference node enters it itself),
  right thumbstick click -> joint position hold (stop), left thumbstick click -> preset HOME.
  Right A is consumed by the headset. While the headset is Paused it re-sends a frozen snapshot,
  so buttons are only detectable while Active. ``--no_mode_events`` disables all three.

DDS outputs are ON by default (``--no_dds`` for a ROS-only session, e.g. with ``--replay_vrs1``).
They move the robot and change its control mode: ROBROS hardware approval and the verification
plan in README.md apply before running against the robot. The controller relay on
``vr/controller`` is state for recorders only (no robot command); the contract and the
publisher class belong to vr_link (``vr_link.messages.teleop_dds``, ``vr_link.network.dds.teleop``),
this producer merely hosts the publisher and counts headset (re)connections for ``session_id``.
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import time
from pathlib import Path

import numpy as np
from general_motion_retargeting.hand_retarget.igris_c_hand_command import HandSourceArbiter
from vr_link.messages.teleop_dds import CONTROLLER_TOPIC as DEFAULT_CONTROLLER_TOPIC
from vr_link.messages.teleop_dds import MOTION_FEEDBACK_TOPIC as DEFAULT_MOTION_FEEDBACK_TOPIC
from vr_link.messages.teleop_dds import MotionFeedbackSample, controller_sample_from_vr_state
from vr_link.network.dds.teleop import ControllerStatePublisher, MotionFeedbackPublisher

from ..dds import (
    DEFAULT_HAND_TOPIC,
    DEFAULT_LOWSTATE_TOPIC,
    DEFAULT_REF_TOPIC,
    DEFAULT_ROBOTSTATE_TOPIC,
    TOPIC_NAMINGS,
    ControlModeClient,
    DdsChannel,
    HandCmdDdsPublisher,
    LowStateSubscriber,
    RefStateDdsPublisher,
    RobotModeSubscriber,
)
from ..retarget.hands import FINGER_SOURCES, MOTOR_IDS, HandTargets
from ..retarget.neck import NeckOverride, head_relative_neck_angles
from ..retarget.retarget_frame import FREE_JOINT_QPOS_SIZE, RetargetFramePayloadBuilder, to_ros_payload
from ..ros.publisher import DEFAULT_NODE_NAME, DEFAULT_ROS_DOMAIN_ID, DEFAULT_ROS_TOPIC, RetargetFrameRosPublisher
from ..shaping.arm_hold import SIDES, ArmHold
from ..shaping.converge import ConvergeGate, low_level_modes_from_sdk
from ..uplink.controller import ConnectionSessions, ControllerEvents
from ..uplink.vrs1_source import DEFAULT_POSE_PORT, Vrs1FileSource, build_human_data, open_live_device, skeleton_complete

logger = logging.getLogger("igris_c_teleop")

DEFAULT_ROBOT = "robros_igris_c_v2"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m igris_c_teleop", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    vr = parser.add_argument_group("VRS1 input")
    vr.add_argument("--pose_host", type=str, default="0.0.0.0", help="Interface the VRS1 listener binds to.")
    vr.add_argument("--pose_port", type=int, default=DEFAULT_POSE_PORT, help="TCP port the headset's VRS1 uplink connects to.")
    vr.add_argument("--stale_timeout", type=float, default=0.5, help="Seconds without a VRS1 record before the device counts as disconnected.")
    vr.add_argument("--replay_vrs1", type=Path, default=None, help="Loop a .vrs1 recording instead of listening for the headset.")

    retarget = parser.add_argument_group("retargeting")
    retarget.add_argument("--robot", type=str, default=DEFAULT_ROBOT, help="GMR target robot.")
    retarget.add_argument("--actual_human_height", type=float, default=1.65, help="Operator height passed to GMR (m).")
    retarget.add_argument("--target_fps", type=float, default=100.0, help="Retarget/publish rate (Hz).")
    retarget.add_argument("--velocity_lpf_hz", type=float, default=8.0, help="First-order LPF cutoff for published velocities; 0 disables.")
    retarget.add_argument("--zero_velocity_frames", type=int, default=10, help="Frames after (re)connection that publish zero velocities.")
    retarget.add_argument("--invert_neck_yaw", action="store_true", help="Negate the HMD-derived neck yaw before publishing.")
    retarget.add_argument("--invert_neck_pitch", action="store_true", help="Negate the HMD-derived neck pitch before publishing.")
    retarget.add_argument(
        "--finger_source",
        choices=FINGER_SOURCES,
        default="auto",
        help="auto (default): per hand, hand-tracking finger curl while tracked, controller grip while the headset "
        "reports the controller tracked (in hand) or once controller input appears, hold otherwise; hand/grip: force "
        "one source; none: no hand command.",
    )

    ros = parser.add_argument_group("ROS 2 output (motion reference)")
    ros.add_argument("--ros_domain_id", type=int, default=DEFAULT_ROS_DOMAIN_ID, help="Must match the inference command.yaml domain_id.")
    ros.add_argument("--ros_topic", type=str, default=DEFAULT_ROS_TOPIC)
    ros.add_argument("--ros_node_name", type=str, default=DEFAULT_NODE_NAME)

    dds = parser.add_argument_group("igris_c_sdk DDS output (hand command, control mode)")
    dds.add_argument("--no_dds", action="store_true", help="Publish the ROS frame only; no HandCmd, no control-mode requests.")
    dds.add_argument("--dds_domain_id", type=int, default=0)
    dds.add_argument(
        "--dds_namespace",
        type=str,
        default=None,
        help="Robot-unit namespace (igris_c configs/robot/*.yaml). Required unless --no_dds: a wrong value fails silently.",
    )
    dds.add_argument("--dds_topic_naming", choices=TOPIC_NAMINGS, default="native", help="Must match the robot's igris_c.network.dds_topic_naming.")
    dds.add_argument("--hand_dds_topic", type=str, default=DEFAULT_HAND_TOPIC, help="Bare HandCmd topic (resolved under the namespace).")
    dds.add_argument("--control_mode_timeout_ms", type=int, default=3000, help="Service timeout per control-mode request.")
    dds.add_argument("--no_mode_events", action="store_true", help="Ignore controller buttons (no control-mode requests).")
    dds.add_argument(
        "--ref_dds_topic",
        type=str,
        default=DEFAULT_REF_TOPIC,
        metavar="TOPIC",
        help=f"Also publish the 31 retargeted joints as MasterArmState on this bare topic (headset hand-overlay display "
        f"input, e.g. {DEFAULT_REF_TOPIC}). Off unless given; the robot bridge does not read it.",
    )
    dds.add_argument(
        "--controller_dds_topic",
        type=str,
        default=DEFAULT_CONTROLLER_TOPIC,
        metavar="TOPIC",
        help=f"Bare topic of the headset controller relay (vr_link teleop_dds Joy_ layout: record_toggle_count, "
        f"session_id, connected, 14 controller axes) read by episode recorders such as the hit-tasks Isaac bridge. "
        f"Default {DEFAULT_CONTROLLER_TOPIC}; published while the skeleton is connected.",
    )
    dds.add_argument("--no_controller_dds", action="store_true", help="Do not relay the headset controller state / recording toggle on DDS.")
    dds.add_argument(
        "--motion_feedback_dds_topic",
        type=str,
        default=DEFAULT_MOTION_FEEDBACK_TOPIC,
        metavar="TOPIC",
        help=f"Bare topic of the motion feedback (vr_link teleop_dds Joy_ layout: converging flag, per-arm hold; the "
        f"progress axis is deprecated and sent as 0) that vr-link-stream turns into Signals for the headset HUD. "
        f"Default {DEFAULT_MOTION_FEEDBACK_TOPIC}.",
    )
    dds.add_argument("--no_motion_feedback_dds", action="store_true", help="Do not publish the motion feedback on DDS.")

    converge = parser.add_argument_group("convergence ramp (reference starts at the robot's measured pose)")
    converge.add_argument("--no_converge", action="store_true", help="Disable the ramp: the reference jumps to the operator as before.")
    converge.add_argument(
        "--converge_seconds", type=float, default=1.0,
        help="Smoothstep blend time from LowState to the live reference (soft start only: the ramp ends as soon as every "
        "joint is within the settle tolerance, and a large gap is bounded by the rate cap).",
    )
    converge.add_argument("--converge_max_rad_s", type=float, default=0.5, help="Per-joint rate cap while converging (0 = no cap).")
    converge.add_argument("--converge_hand_rate", type=float, default=1.0, help="Normalized hand command rate cap (1/s) while converging.")
    converge.add_argument("--lowstate_max_age", type=float, default=0.5, help="A trigger with an older LowState is skipped (seconds).")
    converge.add_argument("--lowstate_dds_topic", type=str, default=DEFAULT_LOWSTATE_TOPIC, help="Bare LowState topic (robot joint positions).")
    converge.add_argument("--robotstate_dds_topic", type=str, default=DEFAULT_ROBOTSTATE_TOPIC, help="Bare RobotState topic (control mode).")

    hold = parser.add_argument_group("arm hold (bare hand untracked)")
    hold.add_argument("--arm_hold", choices=("on", "off"), default="on", help="Freeze an arm's 7 joints while its bare hand is untracked or its body joints are invalid.")
    hold.add_argument("--arm_hold_release_s", type=float, default=0.2, help="Hand must be tracked this long before the hold releases.")
    hold.add_argument("--arm_hold_release_rad_s", type=float, default=0.5, help="Rate cap of the release ramp back to the live reference.")

    parser.add_argument("--measure_fps", action="store_true", help="Print solve rate, IK time and output counters every 2 s.")
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.target_fps <= 0.0:
        raise SystemExit("--target_fps must be positive.")
    if not args.no_dds and not args.dds_namespace:
        raise SystemExit("--dds_namespace is required (robot-unit namespace); pass --no_dds for a ROS-only session.")
    if not args.no_dds and args.finger_source == "none":
        raise SystemExit("DDS hand output needs a finger source other than 'none' (or pass --no_dds).")
    velocity_lpf_hz = args.velocity_lpf_hz if args.velocity_lpf_hz > 0.0 else None

    from general_motion_retargeting import GeneralMotionRetargeting
    from loop_rate_limiters import RateLimiter

    gmr = GeneralMotionRetargeting(
        src_human="xrobot",
        tgt_robot=args.robot,
        actual_human_height=args.actual_human_height,
        verbose=False,
    )
    builder = RetargetFramePayloadBuilder(
        gmr.model,
        robot_name=args.robot,
        target_fps=args.target_fps,
        velocity_lpf_hz=velocity_lpf_hz,
        zero_velocity_frames=args.zero_velocity_frames,
    )
    neck = NeckOverride(
        gmr.model,
        yaw_sign=-1.0 if args.invert_neck_yaw else 1.0,
        pitch_sign=-1.0 if args.invert_neck_pitch else 1.0,
    )
    hands = HandTargets(args.finger_source)
    # Arm hold needs a per-hand source decision even when the finger command is off.
    hold_arbiter = HandSourceArbiter(None) if not hands.enabled else None
    events = ControllerEvents(enabled=not args.no_mode_events)
    arm_hold = ArmHold(
        gmr.model,
        enabled=args.arm_hold == "on",
        release_stable_s=args.arm_hold_release_s,
        release_rate=args.arm_hold_release_rad_s,
    )
    converge = ConvergeGate(
        np.arange(FREE_JOINT_QPOS_SIZE, builder.nq),
        duration_s=args.converge_seconds,
        max_rate=args.converge_max_rad_s,
        hand_rate=args.converge_hand_rate,
        lowstate_max_age_s=args.lowstate_max_age,
        enabled=not args.no_converge and not args.no_dds,
    )
    logger.info(
        "GMR %s: nq=%d dof=%d keybodies=%d; neck override -> qpos%s (ranges %s); fingers=%s; arm hold %s (release after %.2f s, %.2f rad/s); "
        "convergence ramp %s (%.1f s, %.2f rad/s cap, hand %.2f/s, LowState max age %.2f s)",
        args.robot,
        builder.nq,
        builder.dof_count,
        len(builder.body_names),
        list(neck.qpos_addresses),
        [tuple(round(v, 3) for v in r) for r in neck.ranges],
        hands.source,
        "on" if arm_hold.enabled else "off",
        args.arm_hold_release_s,
        args.arm_hold_release_rad_s,
        "on" if converge.enabled else ("off (--no_converge)" if args.no_converge else "off (needs DDS LowState; --no_dds)"),
        args.converge_seconds,
        args.converge_max_rad_s,
        args.converge_hand_rate,
        args.lowstate_max_age,
    )

    # Init order: DDS channel -> service client (before any other reader) -> hand publisher -> ref publisher ->
    # feedback publisher -> LowState / RobotState readers -> ROS.
    channel: DdsChannel | None = None
    control: ControlModeClient | None = None
    hand_dds: HandCmdDdsPublisher | None = None
    ref_dds: RefStateDdsPublisher | None = None
    controller_dds: ControllerStatePublisher | None = None
    feedback_dds: MotionFeedbackPublisher | None = None
    lowstate_dds: LowStateSubscriber | None = None
    mode_dds: RobotModeSubscriber | None = None
    ros: RetargetFrameRosPublisher | None = None
    device = None
    stop = False

    def _request_stop(signum, frame):  # noqa: ARG001
        nonlocal stop
        stop = True

    try:
        if not args.no_dds:
            channel = DdsChannel.create(
                domain_id=args.dds_domain_id, namespace=args.dds_namespace, topic_naming=args.dds_topic_naming
            )
            if events.enabled:
                control = ControlModeClient(channel, timeout_ms=args.control_mode_timeout_ms)
            hand_dds = HandCmdDdsPublisher(channel, args.hand_dds_topic)
            logger.warning(
                "DDS ENABLED (domain %d, namespace %r, naming %s): HandCmd on %s at %.0f Hz (motor ids %s)%s. "
                "These are actuator/control-mode commands; ROBROS hardware approval is required. "
                "Nothing is published while the skeleton stream is lost.",
                args.dds_domain_id,
                args.dds_namespace,
                args.dds_topic_naming,
                hand_dds.resolved_topic,
                args.target_fps,
                list(MOTOR_IDS),
                f"; control-mode requests on {control.request_topic} (right B=low-level, right stick=hold, left stick=HOME)"
                if control is not None
                else "; control-mode requests disabled (--no_mode_events)",
            )
            if args.ref_dds_topic:
                ref_dds = RefStateDdsPublisher(channel, args.ref_dds_topic)
                logger.info(
                    "RefState (31 retargeted joints, MasterArmState type, display only) on %s at %.0f Hz",
                    ref_dds.resolved_topic,
                    args.target_fps,
                )
            if not args.no_controller_dds:
                controller_dds = ControllerStatePublisher(channel.sdk, args.controller_dds_topic)
                logger.info(
                    "Controller relay (headset recording toggle + buttons, Joy_ type, recorder input only) on %s at %.0f Hz",
                    channel.resolve(args.controller_dds_topic),
                    args.target_fps,
                )
            if not args.no_motion_feedback_dds:
                feedback_dds = MotionFeedbackPublisher(channel.sdk, args.motion_feedback_dds_topic)
                logger.info(
                    "Motion feedback (converging, arm hold; Joy_ type, HUD input only) on %s at %.0f Hz",
                    channel.resolve(args.motion_feedback_dds_topic),
                    args.target_fps,
                )
            if converge.enabled:
                converge.low_level_modes = low_level_modes_from_sdk(channel.sdk)
                lowstate_dds = LowStateSubscriber(channel, args.lowstate_dds_topic)
                mode_dds = RobotModeSubscriber(channel, args.robotstate_dds_topic)
                logger.info("Convergence ramp triggers: VRS1 teleop_state RESUMING, robot modes %s on %s", sorted(converge.low_level_modes), mode_dds.resolved_topic)
        else:
            logger.info(
                "DDS disabled (--no_dds): ROS frame only; hand command, controller buttons, RefState, the controller relay, "
                "the motion feedback and the convergence ramp (needs LowState) are off."
            )

        ros = RetargetFrameRosPublisher(domain_id=args.ros_domain_id, topic=args.ros_topic, node_name=args.ros_node_name)
        logger.info("ROS 2 publisher %s on %s (domain %d, RELIABLE depth 1)", args.ros_node_name, args.ros_topic, args.ros_domain_id)

        if args.replay_vrs1 is not None:
            device = Vrs1FileSource(args.replay_vrs1)
            logger.info("Replaying %d VRS1 records from %s", len(device), args.replay_vrs1)
        else:
            device = open_live_device(args.pose_host, args.pose_port, args.stale_timeout)
            logger.info("Listening for the headset pose uplink on %s:%d", args.pose_host, args.pose_port)
        device.start()

        signal.signal(signal.SIGINT, _request_stop)
        signal.signal(signal.SIGTERM, _request_stop)

        rate = RateLimiter(frequency=args.target_fps, warn=False)
        was_connected = False
        sessions = ConnectionSessions()
        last_ros_error_log = 0.0
        last_dds_error_log = 0.0
        window_start = time.monotonic()
        window_count = 0
        window_ik_seconds = 0.0
        last_wait_log = time.monotonic()
        last_teleop_state = None

        while not stop:
            state = device.advance(0.0)
            connected = skeleton_complete(state)
            session_id = sessions.update(connected)
            connected_edge = connected and not was_connected
            if connected != was_connected:
                logger.info("VR skeleton %s", "connected" if connected else "lost; publishing stopped, consumers hold")
                was_connected = connected
            if connected and state.teleop_state != last_teleop_state:
                logger.info("headset teleop_state %s", state.teleop_state.name)
                last_teleop_state = state.teleop_state

            for event in events.update(connected, state):
                if control is not None:
                    control.request(event)
                logger.warning("controller event: %s%s", event, "" if control is not None else " (not sent: DDS off)")

            if not connected:
                builder.reset()
                hands.reset()
                arm_hold.reset()
                converge.reset()
                last_teleop_state = None
                if hold_arbiter is not None:
                    hold_arbiter.reset()
                if time.monotonic() - last_wait_log >= 5.0:
                    # A TCP client may be attached without sending VRS1 records yet:
                    # the headset streams poses only once its start countdown reaches Active.
                    logger.info(
                        "Waiting for fresh VRS1 poses on port %d (no valid record within %.1fs); nothing is published",
                        args.pose_port,
                        args.stale_timeout,
                    )
                    last_wait_log = time.monotonic()
                rate.sleep()
                continue

            now = time.time()
            ik_start = time.perf_counter()
            qpos = gmr.retarget(build_human_data(state), offset_to_ground=True)
            ik_seconds = time.perf_counter() - ik_start

            neck_yaw_pitch = neck.apply(qpos, head_relative_neck_angles(state))
            hand_values = hands.update(state)

            # Reference shaping, in this order: arm hold (per side) then the convergence ramp (all hinges).
            if hands.enabled:
                hand_sources = dict(zip(SIDES, hands.active_sources))
            else:
                hand_sources = {
                    "left": hold_arbiter.decide("left", state.left_hand, state.left_input),
                    "right": hold_arbiter.decide("right", state.right_hand, state.right_input),
                }
            holding = arm_hold.update(
                qpos,
                now=now,
                requests={side: ArmHold.hold_requested(state, side, hand_sources[side]) for side in SIDES},
            )
            mode_sample = mode_dds.latest() if mode_dds is not None else None
            status = converge.update(
                qpos,
                now=now,
                teleop_state=state.teleop_state,
                connected_edge=connected_edge,
                robot_mode=mode_sample[0] if mode_sample is not None else None,
                lowstate=lowstate_dds.latest() if lowstate_dds is not None else None,
            )
            hand_values = converge.shape_hand(hand_values, now)

            if feedback_dds is not None:
                try:
                    feedback_dds.publish(
                        MotionFeedbackSample(
                            converging=status.converging,
                            converge_progress=0.0,  # deprecated axis (vr_link 2026-10-21): the HUD shows the flag only
                            arm_hold_left=holding[0],
                            arm_hold_right=holding[1],
                            stamp_ns=int(now * 1e9),
                        )
                    )
                except Exception as exc:
                    if now - last_dds_error_log > 2.0:
                        logger.warning("Motion feedback DDS publish failed: %s", exc)
                        last_dds_error_log = now
            if hand_dds is not None and hand_values is not None:
                try:
                    hand_dds.publish(hand_values)
                except Exception as exc:
                    if now - last_dds_error_log > 2.0:
                        logger.warning("HandCmd DDS publish failed: %s", exc)
                        last_dds_error_log = now
            if ref_dds is not None:
                # After the neck override so the headset's FK sees the same neck the policy is told.
                try:
                    ref_dds.publish(qpos[FREE_JOINT_QPOS_SIZE:], now)
                except Exception as exc:
                    if now - last_dds_error_log > 2.0:
                        logger.warning("RefState DDS publish failed: %s", exc)
                        last_dds_error_log = now
            if controller_dds is not None:
                # Same wall-clock sample as the RefState stamp; only while connected, like every other output.
                try:
                    controller_dds.publish(
                        controller_sample_from_vr_state(state, session_id=session_id, connected=True, stamp_ns=int(now * 1e9))
                    )
                except Exception as exc:
                    if now - last_dds_error_log > 2.0:
                        logger.warning("Controller relay DDS publish failed: %s", exc)
                        last_dds_error_log = now

            try:
                ros.publish(to_ros_payload(builder.build(qpos, now)))
            except Exception as exc:
                if now - last_ros_error_log > 2.0:
                    logger.warning("ROS frame publish failed: %s", exc)
                    last_ros_error_log = now

            window_count += 1
            window_ik_seconds += ik_seconds
            if args.measure_fps:
                elapsed = time.monotonic() - window_start
                if elapsed >= 2.0:
                    neck_text = (
                        "n/a"
                        if neck_yaw_pitch is None
                        else f"yaw {math.degrees(neck_yaw_pitch[0]):.1f} pitch {math.degrees(neck_yaw_pitch[1]):.1f} deg"
                    )
                    hands_text = ""
                    if hands.enabled and hand_values is not None:
                        left_source, right_source = hands.active_sources
                        hands_text = f", hands L:{left_source} R:{right_source} q_right={np.round(hand_values[:6], 2).tolist()}"
                        if hand_dds is not None:
                            hands_text += f", HandCmd written {hand_dds.published} (rejected {hand_dds.rejected})"
                    if ref_dds is not None:
                        hands_text += f", RefState written {ref_dds.published} (rejected {ref_dds.rejected})"
                    if controller_dds is not None:
                        hands_text += (
                            f", Controller written {controller_dds.published} (rejected {controller_dds.rejected}, "
                            f"session {session_id}, toggle_count {state.record_toggle_count})"
                        )
                    control_text = (
                        f", mode requests ok {control.succeeded} failed {control.failed} dropped {control.dropped}"
                        if control is not None
                        else ""
                    )
                    shaping_text = (
                        f", teleop_state {state.teleop_state.name}, converge "
                        f"{'ACTIVE (blend %.0f%%, residual %.3f rad)' % (100 * status.progress, status.residual) if status.converging else 'idle'}"
                        f" (started {converge.started}, skipped {converge.skipped}), arm hold L:{int(holding[0])} R:{int(holding[1])}"
                    )
                    if lowstate_dds is not None:
                        sample = lowstate_dds.latest()
                        shaping_text += f", LowState {'none' if sample is None else '%.2fs old' % sample[1]}"
                    if mode_dds is not None:
                        sample = mode_dds.latest()
                        shaping_text += f", robot mode {'none' if sample is None else sample[0]}"
                    logger.info(
                        "solve %.1f Hz, IK %.1f ms avg, ROS frames %d (subscribers %d), neck %s%s%s%s",
                        window_count / elapsed,
                        1e3 * window_ik_seconds / max(1, window_count),
                        ros.published,
                        ros.subscription_count(),
                        neck_text,
                        hands_text,
                        control_text,
                        shaping_text,
                    )
                    window_start = time.monotonic()
                    window_count = 0
                    window_ik_seconds = 0.0
            rate.sleep()
    finally:
        # Teardown: input -> service worker -> hand publisher -> ref publisher -> controller relay -> DDS channel -> ROS.
        if device is not None:
            device.close()
        events.enabled = False
        if control is not None:
            control.close()
        if hand_dds is not None:
            hand_dds.close()
        if ref_dds is not None:
            ref_dds.close()
        if controller_dds is not None:
            controller_dds.close()
        if feedback_dds is not None:
            feedback_dds.close()
        if lowstate_dds is not None:
            lowstate_dds.close()
        if mode_dds is not None:
            mode_dds.close()
        if channel is not None:
            channel.release()
        if ros is not None:
            published = ros.published
            ros.close()
            logger.info("Stopped after %d published ROS frames.", published)
    return 0


__all__ = ["DEFAULT_ROBOT", "main", "parse_args"]
