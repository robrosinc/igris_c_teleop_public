# igris_c_gmr

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
```

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
scripts/install_sdk_wheel.sh
scripts/setup.sh            fresh-clone setup: submodules, pixi install, task wheel, install-sdk, check-env
source/igris_c_gmr/         package installed editable; console commands `igris-c-teleop` (session), `igris-c-gmr` (= python -m igris_c_gmr, producer)
  teleop.py                 `igris-c-teleop` / `pixi run teleop`: producer + vr-link-stream as two supervised child processes, one namespace/domain
  producer.py               the producer itself (argparse, 100 Hz loop, init/teardown order); `pixi run producer` runs it alone
  vrs1_source.py            TcpVRDevice wrapper, .vrs1 replay, GMR input conversion
  neck.py                   HMD-relative neck angles, MJCF range clamp
  hands.py                  12-motor normalized hand command (gmr igris_c_hand_command)
  controller.py             button edge detection, event -> control-mode table
  retarget_frame.py         FK keybody + velocities (finite difference, LPF) -> contract frame, JSON serialization
  ros_publisher.py          rclpy String publisher (no spin)
  dds_io.py                 igris_c_sdk channel, HandCmd publisher, control-mode service client (worker)
tests/                      pytest; consumer_oracle.py = port of the C++ parser
thirdparty/{gmr, vr_link, igris_c_sdk}   submodules (vr_link is the former vr_link_schema repo, renamed 2026-09;
                            since vr_link 2.1 it ships no SDK and its stream sender is launched from here)
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
- **This repo** (`source/igris_c_gmr`) and **vr_link** are installed editable from their own
  pyproject.toml. Since vr_link 2.0 has two import roots (`generated/python`, `src/vr_link`),
  re-run `pixi install` after bumping the pin to refresh the editable install.
  The `igris-c-gmr` console command lands in the env's `bin/`.
- Only **gmr** is exposed through `PYTHONPATH` by the activation hook (its setup.py forces
  smplx/opencv/redis). `PYTHONPATH` is written from scratch, so the login shell's
  `igris_c/install` rosidl `igris_c_sdk` never shadows the pybind binding.
- The **igris_c_sdk wheel** is built by a human and installed with pip. Its `.so` links the
  host libssl/libstdc++, but the producer initializes ROS first, so the real process loads
  the env's libssl/libstdc++ first and the SDK reuses them (verified via `/proc/self/maps`).
  It is the only SDK checkout: vr_link 2.1 vendors none and its DDS layer imports the
  wheel from this env.
- **vr-link-stream deps**: conda `av`, `opencv`, `pyyaml` (vr_link's `video` and
  `calibration` features, same bounds) are in this env because `pixi run teleop` starts
  vr_link's VRL2 sender here.

```bash
# Fresh clone in one go: submodules + pixi install + task wheel + install-sdk + check-env
# (plain shell with the host ROS 2 Jazzy sourced; refuses to run under pixi run/shell)
scripts/setup.sh

# Or step by step
git submodule update --init --recursive
pixi install

# igris_c_sdk Python binding wheel (Python 3.12, human-built)
(cd thirdparty/igris_c_sdk && task wheel)     # -> dist/igris_c_sdk-*.whl
pixi run install-sdk

pixi run check-env      # igris_c_sdk and rclpy paths both under .pixi/envs, prints True
pixi run test           # unit / contract tests
pixi run test-ros       # real rclpy round trip
```

`pixi run --clean-env ...` drops all host environment variables but also host commands such
as `bash` and `ldd`, so use it only for Python tasks (`teleop`, `replay`).

**Setup troubleshooting (observed on the first install, 2026-09-18)**

- `task wheel` fails with `No download info given for 'cyclonedds'`: the nested igris_c_sdk
  submodules (cyclonedds, cyclonedds-cxx, pybind11) are empty. Run
  `git submodule update --init --recursive` from the root.
- `pixi run check-env` reports `No module named 'imageio'`: importing the gmr package goes
  through the viewer module. It is in `pyproject.toml` pypi-dependencies, so re-run
  `pixi install`.
- pytest dies with a `launch_testing` plugin error: ROS pytest plugins are autoloading. The
  `pixi run test` / `test-ros` tasks already set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
- Re-running `pixi install` keeps the pip-installed SDK wheel (it survived the env re-solve
  for the RoboStack switch on 2026-09-21).
- The editable install creates `source/igris_c_gmr.egg-info/`. It is in `.gitignore`.

**Binding pitfall**: an old build's `.so` registered the DDS type name as
`igris_c_sdk::msg::dds_::HandCmd_`, silently mismatching the robot bridge
(`igris_c::msg::dds::HandCmd`). Always rebuild and install from the submodule pin, and check
that `strings <so> | grep -o 'igris_c.*::HandCmd'` contains no `dds_`.
If `pixi run check-env` prints "SHADOWED by rosidl package", the igris_c overlay is still on
PYTHONPATH.

**Service topic prefix pitfall (observed 2026-09-18)**: the `IgrisC_Client` of the
2026-09-15 wheel left in `igris_c_teleop_env` (built from igris_c 2ba902e) writes requests to
`rt/service/control_mode/request`. The submodule pin (6b40103) and the current igris_c bridge
(`dds_manager.cpp`) use `service/control_mode/request`. With the old wheel, button requests
never reach the bridge and the failure is silent (timeout only). The producer logs the loaded
SDK version and path at startup; confirm it is the wheel built from the submodule. The
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
`igris_c.network.dds_topic_naming`), `--hand_dds_topic handcmd`, `--no_mode_events`,
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
robot command).

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

Rollback: the hit-bench Redis + `run_ros2_redis_bridge.sh` path is still in place.

## Open items (human confirmation)

- igris_c bridge/SDK version and topic naming on the robot NUC. The wire of submodule pin
  `igris_c_sdk@6b40103` and `igris_c` workspace `78bec2c` is identical (the only difference is
  one test snapshot).
- Confirm once that `pixi install` keeps the pip-installed SDK wheel.
- The finger normalization -> joint angle mapping is a linear placeholder (gmr
  `igris_c_hand_command`); hardware bend ratios not yet measured.
