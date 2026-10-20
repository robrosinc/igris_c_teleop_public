# igris_c_teleop

VR headset (vr_system, VRS1) teleoperation **producer**. It retargets the human skeleton
onto IGRIS-C with GMR, publishes the ROS 2 motion-reference topic that
`igris_c_inference_node` subscribes to directly, and sends finger commands and controller
buttons (control-mode switches) straight to the robot through igris_c_sdk (CycloneDDS).
It is a port of hit-bench's `scripts/teleoperation/vr_system_teleop.py` without the
intermediate Redis stage and gmr's `redis_to_ros2_bridge.py`.

> DRAFT. ROBROS AI governance: the DDS output of this producer moves a real robot and
> switches its control mode. Before connecting to the robot, a hardware-change approval and
> a human-performed record of the verification procedure below are required.

## Data flow

```
headset (vr_system, VRS1 FlatBuffers over TCP -> this host :9102)
  -> vr_link.TcpVRDevice (gmr joint convention)
  -> GeneralMotionRetargeting(xrobot -> robros_igris_c_v2)          38 qpos = root 7 + 31 hinges
  -> NeckOverride                                                   HMD-relative neck yaw/pitch into qpos
  -> ArmHold                                                        7 arm joints frozen while that bare hand is untracked
  -> ConvergeGate (DDS lowstate + robotstate, VRS1 teleop_state)    31 hinges ramped from the robot's measured pose to the
                                                                    operator on Start / Resume / low-level entry
  -> RetargetFramePayloadBuilder -> to_ros_payload (JSON)
  -> ROS 2 std_msgs/String  /gmr/teleop/retarget_frame  (domain 91, RELIABLE depth 1)
       -> igris_c_inference_node  motion term (23 policy joints -> ONNX WBC -> DDS lowcmd)
                                  motion_command term (6 wrist + 2 neck -> direct joint targets)
  -> HandTargets -> HandCmdDdsPublisher     DDS <ns>/handcmd  (12 normalized motors, 100 Hz)
  -> ControllerEvents -> ControlModeClient  DDS <ns>/service/control_mode/request
  -> RefStateDdsPublisher (opt-in)          DDS <ns>/gmr/refstate  (31 retargeted joints, MasterArmState type, 100 Hz,
                                            display-only for the headset hand overlay - the robot bridge does not subscribe)
  -> ControllerStatePublisher (vr_link)     DDS <ns>/vr/controller  (headset UI record-toggle counter, session id, 14 controller
                                            axes; Joy_ type, 100 Hz, only while the skeleton is connected. Input for episode
                                            recorders (hit-tasks Isaac bridge etc.) only - the robot bridge does not subscribe)
  -> MotionFeedbackPublisher (vr_link)      DDS <ns>/teleop/motion_feedback  (converging flag + progress, per-arm hold; Joy_ type,
                                            100 Hz while the skeleton is connected. Read by vr-link-stream into Signals for the
                                            headset HUD only - the robot bridge does not subscribe)
```

### Reference shaping (convergence ramp, arm hold)

Both run after GMR and before any output, on the 38-qpos reference, so the ROS frame,
`gmr/refstate` and the hand command all see the shaped values; the inference node and the
robot bridge are untouched.

- **Convergence ramp** (`shaping/converge.py`, `shaping/ramp.py`; on by default, `--no_converge` off). The
  inference node copies the reference into `lowcmd` without any rate limit, so a Start or
  Resume with the operator away from the robot's current pose would command a jump. The
  gate subscribes to DDS `lowstate` (31 joint positions) and `robotstate` (control mode)
  and, on a trigger, replaces the 31 hinge joints of the reference with a smoothstep blend
  from the LowState pose to the live reference (`--converge_seconds 2.0`), with a per-joint
  rate cap (`--converge_max_rad_s 0.5`) that extends the ramp when the distance is large;
  the root pose is never touched. The hand command is rate-limited meanwhile
  (`--converge_hand_rate 1.0`/s). Triggers: VRS1 `teleop_state` entering `Resuming`
  (vr_system Start and Resume), the robot mode entering the low-level family
  (`ROBOT_STATE_LOW` / `ROBOT_STATE_WALK_LOW`, read from the SDK binding) from a known other
  mode, and, for a headset without the field (`Unknown`), the skeleton-connection edge. A
  trigger without a LowState sample fresher than `--lowstate_max_age 0.5` s is skipped with
  a warning (the reference then jumps as before). `--no_dds` disables the ramp (no LowState).
- **Arm hold** (`shaping/arm_hold.py`; `--arm_hold on|off`, default on). With bare hands (no
  controller) the PICO body tracker keeps sending wrist/elbow/shoulder joints after the hand
  leaves the headset's view, and GMR would follow them. When the finger arbiter decides
  `hold` for a hand (hand untracked, controller neither tracked nor active) or any of that
  arm's body joints arrives `valid=false`, the 7 joints of that arm are frozen at the reference of the tick the hold
  engaged (so the arm keeps its configuration relative to the torso). Once the hand is tracked
  again for `--arm_hold_release_s 0.2` s, the held pose ramps to the live reference at
  `--arm_hold_release_rad_s 0.5` (smoothstep 1 s). A controller in the hand never holds:
  the headset reports `ControllerInput.tracked` (vr_link 2.1, from the runtime's controller
  tracking status) and the arbiter treats a tracked controller as `grip` even when idle, so
  the arm follows the controller through body tracking. Headsets without the field fall back
  to "controller input seen", which holds until the first button press (the pre-2.1
  behaviour). Per side, independent; fingers already
  hold through `HandSourceArbiter`.
- **Feedback** (`--motion_feedback_dds_topic teleop/motion_feedback`,
  `--no_motion_feedback_dds`): `MotionFeedbackSample(converging, converge_progress,
  arm_hold_left, arm_hold_right)` every tick while connected. `vr-link-stream` folds it into
  the VRL2 `Signals` record; the headset HUD shows `CONVERGING xx%` (pill, popup ring) and an
  arm-hold toast, also in NO CAMERA mode (the sender heartbeats `Signals` without video).

The ROS topic carries **only the whole-body motion frame**. Keys: `fps, root_pos,
root_rot(wxyz), root_vel, root_angvel, dof_pos, dof_vel (MJCF order, 31), keybody_pos_local,
keybody_rot_local, link_body_list, record_meta`. The consumer parser
(igris_c_inference_public `MotionReferenceSource.cpp`) requires `dof_pos, dof_vel, root_rot,
link_body_list, keybody_rot_local` and ignores the rest. `tests/consumer_oracle.py` is a
Python port of that parser used by the contract tests.

Controller button mapping (one rising edge, only while the skeleton is connected; identical
to the retired `igris_c_gmt_public` `gmr_control_mode_dds_bridge`):

| Button | Event | DDS control-mode request |
|---|---|---|
| Right B | `lowcmd_start` | `CONTROL_MODE_CMD_LOW_LEVEL_JOINT_CONTROL` (redundant: the inference node enters it by itself) |
| Right thumbstick click | `stop` | `CONTROL_MODE_CMD_JOINT_POSITION_HOLD` |
| Left thumbstick click | `go_home` | `CONTROL_MODE_CMD_MOTION_PRESET`, preset `HOME` |

Right A is consumed by the headset state machine. While the headset is Paused it resends a
frozen snapshot, so buttons are only detected in the Active state. Service calls block on a
worker thread (default 3 s timeout) and never stall the 100 Hz loop. When the queue (4) is
full, new requests are dropped with a warning.

## Layout

```
pyproject.toml              pixi env (incl. RoboStack ROS) + tasks + package metadata + pytest config
scripts/activate_ros.sh     pixi activation hook: PYTHONPATH=thirdparty/gmr only (ROS, source and vr_link are installed in the env)
scripts/setup.sh            fresh-clone setup: submodules (gmr, vr_link), pixi install, check-env
source/igris_c_teleop/      package installed editable; console commands `igris-c-teleop` (session), `igris-c-gmr` (= python -m igris_c_teleop, producer).
                            Sub-packages follow the data flow (headset -> retarget -> shaping -> outputs); imports only go downstream.
  app/                      entry points
    producer.py             the producer itself (argparse, 100 Hz loop, init/teardown order); `pixi run producer` runs it alone
    teleop.py               `igris-c-teleop` / `pixi run teleop`: producer + vr-link-stream as two supervised child processes, one namespace/domain
  uplink/                   headset -> producer (VRS1)
    vrs1_source.py          TcpVRDevice wrapper, .vrs1 replay, GMR input conversion
    controller.py           button edge detection, event -> control-mode table, ConnectionSessions
  retarget/                 GMR output -> reference
    neck.py                 HMD-relative neck angles, MJCF range clamp
    hands.py                12-motor normalized hand command (gmr igris_c_hand_command)
    retarget_frame.py       FK keybody + velocities (finite difference, LPF) -> contract frame, JSON serialization
  shaping/                  reference shaping (safety-relevant: the only rate limits before the robot)
    ramp.py                 JointRamp (smoothstep + rate cap on selected qpos), RateLimiter
    converge.py             ConvergeGate: Start/Resume/low-level-entry triggers, ramp from LowState, hand rate limit
    arm_hold.py             ArmHold: per-arm freeze while the bare hand is untracked, debounced release ramp
  ros/
    publisher.py            rclpy String publisher (no spin)
  dds/                      igris_c_sdk side; `from igris_c_teleop import dds` re-exports everything
    channel.py              lazy SDK import, struct lookup, DdsChannel (ChannelFactory wrapper), topic naming
    hand_cmd.py             HandCmdDdsPublisher (robot command)
    ref_state.py            RefStateDdsPublisher (opt-in, headset display only)
    state_readers.py        LowStateSubscriber, RobotModeSubscriber (ramp inputs, read-only)
    control_mode.py         ControlModeClient (service requests on a worker thread; robot command)
tests/                      pytest; consumer_oracle.py = port of the C++ parser
thirdparty/{gmr, vr_link}   submodules (vr_link is the former vr_link_schema repo, renamed 2026-09;
                            since vr_link 2.1 it ships no SDK and its stream sender is launched from here).
                            igris_c_sdk is not a submodule any more (2026-10-16): pixi installs its release wheel.
```

## Environment (pixi)

One process needs GMR (mujoco/mink), rclpy and igris_c_sdk together. Everything lives in a
single env, `.pixi/envs/default`; the host `/opt/ros` is not used (switched 2026-09-21).

- **ROS 2 Jazzy** comes from RoboStack (`robostack-jazzy`) conda packages. Channel order and
  version bounds match the `core` feature of `~/workspace/repos/igris_c` `pyproject.toml`,
  the env the robot NUC runs (`rclpy >=7.1.9,<8`, `std-msgs >=5.3.6,<6`,
  `rmw-fastrtps-cpp >=8.4.3,<9`). RoboStack `.so` files use `RPATH`, so even when the login
  shell's `LD_LIBRARY_PATH` (igris_c install, /opt/ros) is inherited, the env's `librcl.so`
  wins (verified with ldd).
- **This repo** (`source/igris_c_teleop`) and **vr_link** are installed editable from their own
  pyproject.toml. Since vr_link 2.0 has two import roots (`generated/python`, `src/vr_link`),
  re-run `pixi install` after bumping the pin to refresh the editable install.
  The `igris-c-gmr` console command lands in the env's `bin/`.
- Only **gmr** is exposed through `PYTHONPATH` by the activation hook (its setup.py forces
  smplx/opencv/redis). `PYTHONPATH` is written from scratch, so the login shell's
  `igris_c/install` rosidl `igris_c_sdk` never shadows the pybind binding.
- The **igris_c_sdk wheel** is a pypi URL dependency in `pyproject.toml`: the release wheel
  of `github.com/robrosinc/igris_c_sdk_public` (`dist/`), pinned by release tag (v2.1.1 since
  2026-10-16; before that a human-built wheel from a `thirdparty/igris_c_sdk` submodule).
  `pixi install` downloads it; `pixi.lock` records the URL and version but no content hash,
  so the release tag is the only pin (`git describe`-style immutability is not enforced; a
  re-tagged release would be picked up silently). The wheel is self-contained
  (CycloneDDS statically linked); its `.so` links the host libssl/libstdc++, but the producer
  initializes ROS first, so the real process loads the env's libssl/libstdc++ first and the
  SDK reuses them (verified via `/proc/self/maps`). No SDK source is checked out here:
  vr_link 2.1 vendors none and its DDS layer imports the wheel from this env.
- **vr-link-stream deps**: conda `av`, `opencv`, `pyyaml` (vr_link's `video` and
  `calibration` features, same bounds) are in this env because `pixi run teleop` starts
  vr_link's VRL2 sender here.

```bash
# Fresh clone in one go: submodules + pixi install + check-env
scripts/setup.sh

# Or step by step
git submodule update --init --recursive
pixi install            # conda + PyPI deps incl. the igris_c_sdk release wheel (tag-pinned URL)

pixi run check-env      # igris_c_sdk and rclpy paths both under .pixi/envs, prints True
pixi run test           # unit / contract tests
pixi run test-ros       # real rclpy round trip
```

**Updating the SDK wheel**: pick the release tag in `igris_c_sdk_public` (its release commit
message names the internal `igris_c_sdk` commit it was built from), change the tag and the
wheel filename together in the `igris_c_sdk` URL of `pyproject.toml`, run `pixi install`,
then `check-env`, `test`, and steps 1, 6 and 7 of the verification procedure below before the
wheel talks to a robot.

`pixi run --clean-env ...` drops all host environment variables but also host commands such
as `bash` and `ldd`, so use it only for Python tasks (`teleop`, `replay`).

**Setup troubleshooting (observed on the first install, 2026-09-18)**

- `pixi run check-env` reports `No module named 'imageio'`: importing the gmr package goes
  through the viewer module. It is in `pyproject.toml` pypi-dependencies, so re-run
  `pixi install`.
- pytest dies with a `launch_testing` plugin error: ROS pytest plugins are autoloading. The
  `pixi run test` / `test-ros` tasks already set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
- The editable install creates `source/igris_c_teleop.egg-info/`. It is in `.gitignore`.
- The repository and package were renamed `igris_c_gmr` -> `igris_c_teleop` (2026-10-15).
  Existing clones: `git remote set-url origin git@bitbucket.org:robrosinc/igris_c_teleop.git`,
  then `pixi install` to refresh the editable install and remove `source/igris_c_gmr.egg-info/`.

**Binding pitfall**: the DDS type name of the binding must match the bridge exactly, and a
mismatch is silent. The current bridge (`igris_c_bridge/src/dds/dds_manager.cpp`, igris_c
workspace bed8554, read 2026-10-16) subscribes to `igris_c_sdk::msg::dds_::HandCmd_` on the
bare `handcmd` topic; the v2.1.1 wheel registers that same name (its IDL is identical to the
earlier 6b40103 pin). An earlier version of this paragraph demanded that
`strings <so> | grep -o 'igris_c.*::HandCmd'` contain no `dds_`; that criterion contradicts
the current bridge source and was dropped on 2026-10-16 (human re-check requested). Always
install the tag-pinned release wheel from `pyproject.toml` (`pixi install`) and compare the
name printed by `strings <so> | grep -o 'igris_c_sdk::msg::dds_::HandCmd_'` with the bridge
build that the robot NUC actually runs.
If `pixi run check-env` prints "SHADOWED by rosidl package", the igris_c overlay is still on
PYTHONPATH.

**Service topic prefix pitfall (observed 2026-09-18)**: the `IgrisC_Client` of the
2026-09-15 wheel left in `igris_c_teleop_env` (built from igris_c 2ba902e) writes requests to
`rt/service/control_mode/request`. The pinned release wheel (v2.1.1, like the earlier
submodule pin 6b40103) and the current igris_c bridge (`dds_manager.cpp`) use
`service/control_mode/request`. With the old wheel, button requests never reach the bridge
and the failure is silent (timeout only). The producer logs the loaded SDK version and path at
startup; confirm the version is the pinned release and the path is under `.pixi/envs`. The
`handcmd` topic of `HandCmd` is the same in both versions.

## Start order (robot NUC)

1. igris_c controller/bridge (DDS `lowstate`, `lowcmd`, `handcmd`, control-mode service),
   started by a human.
2. The teleop session (producer + headset video downlink), or the producer alone:
   ```bash
   pixi run teleop -- --dds_namespace igris_c_<unit> --target robot --measure_fps    # producer + vr-link-stream
   pixi run producer -- --dds_namespace igris_c_<unit> --measure_fps                 # producer only (dev/debug)
   ```
   `teleop` starts `vr-link-stream` (vr_link's VRL2 sender: camera, calibration and
   frame-matched robot state on ports 9100/9101) and the producer as two child
   processes of this env with the same namespace/domain, forwards Ctrl-C to both and
   stops the survivor when one exits. `--target` selects the packaged vr_link config:
   `robot` (igris_c_camera DDS topic + the measured calibration under
   `thirdparty/vr_link/data/calibration/`), `usb` (stereo camera on this PC, same
   calibration), `mujoco`, `isaac` (ideal calibrations) or `shm`. Unknown arguments go
   to the producer; `--no_stream` equals `producer`;
   `--dry_run` prints both command lines. A non-native `--dds_topic_naming` needs
   `--stream_config` (the sender reads `dds.topic_naming` from its file only).
   `--head_rgb on|off` forwards vr_link's `--head-rgb`: the second video stream of the
   head colour camera (`VideoFrame.stream_id = HeadRgb`, vr_link 2.2) on top of the
   packaged config's `head_rgb.enable` (off); only for headsets running a vr_system
   build that demultiplexes `stream_id`.
   Before 2026-10-15 `pixi run teleop` meant the producer alone; that is `producer` now.
   Put the headset in Active and confirm `VR skeleton connected` in the log.
3. `ros2 run igris_c_inference_public igris_c_inference_node --config <policy>/params.yaml`
   (command.yaml `source.params.domain_id` is 91, topic `/gmr/teleop/retarget_frame`).

**Order matters.** The inference node exits if it does not receive 46 frames within 5 s of
startup. This producer publishes nothing without a skeleton (no default-pose seeding as in
the old bridge), so start step 3 while step 2 is streaming. During operation, a disconnect is
covered by the inference node's `on_stale: hold_last`, which keeps the last reference, and
publishing resumes on reconnect.

Main options: `--no_dds` (ROS only, no hand/button output), `--replay_vrs1 <file>` (replay a
recording without a headset), `--finger_source auto|hand|grip|none`,
`--dds_topic_naming native|ros_compatible` (must match the robot's
`igris_c.network.transport`: `cyclonedds_native` pairs with `native`,
`cyclonedds_ros_compatible` with `ros_compatible`; SDK v2.1.1 replaced the earlier
`igris_c.network.dds_topic_naming` key), `--hand_dds_topic handcmd`, `--no_mode_events`,
`--ros_domain_id 91`, `--invert_neck_yaw/--invert_neck_pitch`,
`--ref_dds_topic gmr/refstate` (also publish the 31 retargeted joints as `MasterArmState`;
input for the hand-target overlay of vr_system `ARobotTeleopPawn`. Not published unless
given. Contract: vr_system `docs/Arm_Impedance_Overlay_Plan.md`),
`--controller_dds_topic vr/controller` / `--no_controller_dds` (headset controller relay:
re-publishes the VRS1 `record_toggle_count` together with a `session_id` and the controller
axes as `Joy_`. On by default. Layout and publisher class are owned by vr_link
`messages/teleop_dds.py` and `network/dds/teleop.py`; this producer only hosts them and
counts reconnections (`ConnectionSessions`). The recorder toggles by `(new - last) % 2`
parity and re-seeds when `session_id` changes, so a reconnect never flips recording. Not a
robot command), the reference-shaping options above (`--no_converge`, `--converge_seconds`,
`--converge_max_rad_s`, `--converge_hand_rate`, `--lowstate_max_age`, `--lowstate_dds_topic
lowstate`, `--robotstate_dds_topic robotstate`, `--arm_hold on|off`, `--arm_hold_release_s`,
`--arm_hold_release_rad_s`, `--motion_feedback_dds_topic teleop/motion_feedback`,
`--no_motion_feedback_dds`).

The DDS namespace has no default. A wrong namespace reaches nothing without any error, so
compare the resolved topic names in the startup log (`<ns>/handcmd`,
`<ns>/service/control_mode/request`) with the names the robot bridge subscribes to. The
current igris_c bridge source (`igris_c_bridge/src/dds/dds_manager.cpp`) subscribes to the
bare `handcmd` in native naming. If the robot NUC runs a different bridge version (old `rt/`
prefix), match it with `--hand_dds_topic rt/handcmd`.

## Verification procedure (human-performed, pass criteria)

1. `pixi run check-env`: igris_c_sdk path under `.pixi/envs`, prints `True`.
2. `pixi run test` all pass, `pixi run test-ros` passes.
3. Without a headset: `pixi run replay -- <file>.vrs1`. In another ROS shell,
   `ROS_DOMAIN_ID=91 ros2 topic hz /gmr/teleop/retarget_frame` ≈ 100 Hz,
   `ros2 topic echo --once` shows 31 `dof_pos` entries and `link_body_list`,
   `ros2 topic info -v` shows RELIABLE / KEEP_LAST(1).
4. Attach the inference node (torque OFF or simulation) to the replay: ready log within 5 s,
   no `invalid motion-reference payload`. Stop the producer -> consumer holds, restart ->
   resumes.
5. Repeat 3 and 4 with a real headset connection. Publishing stops on disconnect and resumes
   on reconnect.
6. DDS hand path (torque OFF, approval recorded): resolved topic names in the startup log
   match the bridge's subscription names. With an SDK `HandState`/`handcmd` subscriber:
   12 motors, ids 11-16/21-26, q∈[0,1], 100 Hz, 0 Hz without a skeleton.
7. Control mode (approval required, safe posture): right B / right stick / left stick once
   each -> one request per press in the log and `accepted`; robot mode switch confirmed; loop
   period holds during requests with `--measure_fps`. No re-fire while held down.
8. Controller relay (no robot effect): subscribe to `<ns>/vr/controller` with vr_link
   `ControllerStateSubscriber`; 100 Hz while the skeleton is connected, 0 Hz when it drops,
   `frame_id vr_link.controller.v1`. Each headset UI record toggle increments
   `record_toggle_count` by 1, each headset reconnect increments `session_id` by 1 (compare
   with `session` and `toggle_count` in the `--measure_fps` log). In the recorder (hit-tasks
   bridge `--record`), one toggle gives `recording started`, the next `stopped`, and no toggle
   right after a reconnect.
9. Ctrl-C: clean shutdown, no DDS/rmw abort, exit 0.
10. Convergence ramp (simulation first, then torque ON only with approval and a safe
    posture, hands clear). Headset in Paused with the operator's arms far from the robot's,
    then Resume: the log shows `convergence ramp started (teleop_state RESUMING ...)`, the
    `--measure_fps` line shows `converge ACTIVE xx%` reaching `idle` after ~2 s (longer when
    the 0.5 rad/s cap binds); the robot moves smoothly, no jump, `ros2 topic echo` of
    `dof_pos` shows a monotone blend. Same when the robot enters low-level control while the
    headset is Active (`robot entered low-level control` trigger). With the producer started
    before the robot bridge (no LowState), the trigger is logged as skipped and the reference
    jumps as before. Headset HUD: pill `CONVERGING xx%` in cyan and the center ring; in NO
    CAMERA mode as well. Pass: no visible jump at Start/Resume/low-level entry; the ramp
    ends within `converge_seconds` + distance / cap.
11. Arm hold (bare hands, torque OFF first). Cover one hand: the log shows `left arm HELD`,
    the HUD toast names the arm, the robot arm stays still while the torso and the other arm
    follow. Uncover: after 0.2 s `released`, the arm ramps back at <= 0.5 rad/s, toast gone.
    Rapid cover/uncover within 0.2 s neither releases nor re-engages. Pass: no arm motion
    while held, no jump on release.

Rollback: `--no_converge --arm_hold off --no_motion_feedback_dds` restores the pre-2026-10
producer behavior (reference jumps; arms follow the body tracker); the hit-bench Redis +
`run_ros2_redis_bridge.sh` path is still in place.

## Open items (human confirmation)

- igris_c bridge/SDK version and topic naming on the robot NUC. The wire of the earlier
  submodule pin `igris_c_sdk@6b40103` and `igris_c` workspace `78bec2c` was identical (the only
  difference is one test snapshot). The release wheel v2.1.1 used since 2026-10-16 is built
  from internal `igris_c_sdk` 8bc8c7b; between 78bec2c and that commit only
  `include/igris_c_sdk/namespace_resolver.hpp` (25 lines) and the release-prep commit differ,
  msg/srv/idl are unchanged. The `namespace_resolver.hpp` change still needs a human read
  (topic name resolution) before the wheel talks to a robot.
- The finger normalization -> joint angle mapping is a linear placeholder (gmr
  `igris_c_hand_command`); hardware bend ratios not yet measured.
- Controller grip -> `handcmd` is linear over the full 0..1 range since 2026-10-21
  (`retarget/hands.py` rescales away gmr's internal `GRIP_FULL_COMMAND` cap of 0.6; grip 0.5
  publishes 0.5). Hand tracking was always 0..1. The physical hand at command 1.0 from a full
  grip squeeze has not been validated yet.
