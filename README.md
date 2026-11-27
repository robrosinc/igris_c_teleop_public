# igris_c_teleop

VR teleoperation producer for the **IGRIS-C** humanoid.

The VR headset (`vr_system`) streams the operator's full-body skeleton, head pose, hands and
controller state to this host over TCP (`vr_link`). This producer retargets the human motion
onto the IGRIS-C robot with [GMR](https://github.com/YanjieZe/GMR), shapes the result so the
robot never receives a sudden jump, and publishes the reference motion as a ROS 2 topic for the
whole-body policy (`igris_c_inference_node`). Hand commands and control-mode switches go to
the robot directly over DDS (`igris_c_sdk`).

> The DDS output of this producer moves a real robot and switches its control mode.
> Validate on a simulator or with torque off before connecting to hardware.

## Pipeline

```
VR headset (vr_system)
   │  skeleton, head pose, hands, controller   (TCP, vr_link)
   ▼
igris_c_teleop
   ├─ retarget      GMR: human skeleton -> IGRIS-C joints, neck from head pose
   ├─ shaping       resume anchor, convergence ramp, arm hold (see below)
   └─ outputs
        ├─ ROS 2  reference motion frame  ──>  igris_c_inference_node  ──>  robot (lowcmd)
        ├─ DDS    hand command            ──>  robot
        ├─ DDS    walk velocity (cmd_vel) ──>  robot   (thumbsticks; consumed in walk mode)
        ├─ DDS    control-mode request    ──>  robot   (controller buttons)
        └─ DDS    motion feedback         ──>  headset HUD (via vr-link-stream)

vr-link-stream (vr_link)   robot camera + robot state  ──>  headset   (started by `pixi run teleop`)
```

Source layout follows the pipeline (`source/igris_c_teleop/`): `uplink` (headset input),
`retarget`, `shaping`, `ros` / `dds` (outputs), `app` (entry points).

## Reference timing

The headset sends pose records at its own frame rate (about 52 Hz with 13 to 28 ms gaps in a
2026-11 capture), below the 100 Hz loop (`--target_fps`). Since 2026-11-26 the reference frame
follows the headset, not the loop (`--no_source_timing` restores the previous behaviour):

- A record is retargeted and published **once**. A loop tick that finds no new record publishes
  no reference frame, so `/gmr/teleop/retarget_frame` runs at the headset's record rate.
- Reference velocities are differences over the **headset timestamps** of two records, not over
  the loop period.
- `record_meta` carries `session_id` (one per producer run), `source_timestamp_ns` (the record's
  headset clock) and `source_session_id` (one per continuous headset clock: it changes on every
  reconnection and when that clock steps backwards). The inference node
  (`igris_c_inference_public`) places the frames on its history timeline by
  `source_timestamp_ns` and resamples them at the policy rate.
- The DDS outputs (hand command, `cmd_vel`, controller relay, motion feedback) keep the loop rate.
- A headset that sends no new record for longer than the inference node's `stale_after_ms`
  (default 100) makes the node hold its reference and, on the next record, rebuild its history
  and re-align the heading to the robot. Before, the producer kept publishing the last pose until
  `--stale_timeout` (0.5 s) and the node saw no interruption.

A sender without `timestamp_ns` (0) is handled as before: every tick, host clock.

## Reference shaping

The policy applies the reference without any rate limit of its own, so these stages are the
only thing between the operator and a joint jump. All are enabled by default.

- **Convergence ramp** (`--no_converge` to disable). When teleoperation starts or resumes, or
  the robot enters low-level control, the reference is blended from the robot's measured joint
  pose to the operator's pose instead of jumping. The ramp is rate-capped (default 0.6 rad/s) and
  ends as soon as every ramped joint is close to the live reference.
  Since 2026-11-26 the ramp shapes the **upper body only** (`--converge_joints upper`, the
  default): shoulders, elbows, wrists and neck, 16 of the 31 joints. The waist and leg reference
  (15 joints) and the root pose follow the operator from the first frame: a posture change of
  the lower body made while paused reaches the policy as a step. `--converge_joints all` ramps
  every joint as before.
  Each ramp start or restart resets reference velocity estimation to avoid differentiating
  across the switch to measured joints. Root and joint velocities are zero for
  `--zero_velocity_frames` frames afterward (default 10: about 0.2 s at a headset record rate near
  50 Hz, 0.1 s with `--no_source_timing`).
  A ramp always starts from a fresh `LowState` (`--lowstate_max_age`, default 0.5 s). When none
  is there at that moment the producer waits: it holds the last reference and hand command it
  published (on a first start it publishes nothing) and starts the ramp when one arrives; the
  log repeats `convergence ramp waiting` meanwhile.
  When the ramp of a Start or Resume has finished, the producer reports that request's
  `resume_count` back to the headset (`teleop/motion_feedback` -> `vr-link-stream` ->
  `Signals.converged_resume_count`). A headset build that waits for this acknowledgement stays
  in Resuming until then and reaches Active only with `vr-link-stream` and the motion feedback
  running (not with `--no_dds` or `--no_motion_feedback_dds`).
  The joints move in step: one blend sized by the largest gap, so that this joint peaks at the
  rate cap and all others reach the same fraction of their own gap at the same time (a large
  posture change therefore takes longer than the cap alone would: about 3.7 s for a 84 deg elbow
  gap at the default cap). `--converge_per_joint` restores the fixed `--converge_seconds` blend
  with each joint capped on its own, where joints with small gaps arrive first and the body
  passes through postures that lie on neither side of the change.
  Root pose: with the default `--converge_joints upper` no leg joint is ramped, so the root
  (torso tilt and height) is the operator's live value throughout. With `--converge_joints all`
  the root is recomputed while the ramp runs from the ramped joints with the feet kept where the
  live reference has its feet, so it starts at the pose the robot is in and ends on the
  operator's root together with the joints; `--no_converge_root` then leaves the root at the
  operator's live value instead (it steps at the resume by the full tilt and height difference).
- **Resume anchor** (`--no_resume_anchor` to disable). A turn or a walk made while paused is
  not passed to the retargeting: on every Start/Resume the operator's skeleton is rotated about
  the vertical axis and shifted horizontally, once, so that it continues with the heading (hip
  line) and foot position of the last skeleton that was retargeted. Without it the reference
  yaw and position step in one tick; the inference node holds the heading, but it rotates its
  whole reference history by the current correction, so the frames from before the step read as
  turned by the full angle until they leave the history window. Posture differences (joints,
  tilt, height) are not touched here; the convergence ramp handles them. The log reports
  `operator frame re-anchored at resume` with the turn and the distance.
- **Arm hold** (`--arm_hold off` to disable). When a bare hand leaves the headset's tracking
  view, that arm is frozen at its last pose instead of following stale body-tracking data.
  Once the hand is tracked again it ramps back smoothly. A hand holding a controller never
  holds.
- **Motion feedback** (`--no_motion_feedback_dds` to disable). The converging and arm-hold
  states are published for the headset HUD so the operator can see why the robot is not
  following.

## Controller buttons

| Button | Robot control mode |
|---|---|
| Right B | low-level joint control (teleoperation) |
| Right thumbstick click | joint position hold |
| Left thumbstick click | motion preset HOME |

Right A is consumed by the headset itself. Buttons are only read while the headset is Active.

## Walk velocity (thumbsticks)

While the headset is Active the thumbsticks are published as a `cmd_vel` (`TwistStamped`) on DDS,
which the robot consumes in its walk mode (Low Level Walkmode) and ignores in low-level joint
control. Layout and numbers are the robot gamepad's (igris_c `joystick_publisher_node.cpp`), so
VR and gamepad feel the same. `--no_walk_dds` disables it.

| Thumbstick | Command |
|---|---|
| Left forward / back | walk forward / backward (`vx`) |
| Left left / right | step left / right (`vy`) |
| Right left / right | turn left / right (yaw rate) |

Full deflection gives `--walk_max_linear` (default 0.5 m/s) and `--walk_max_yaw_rate` (default
0.8 rad/s); the deadzone (`--walk_deadzone`, default 0.1) is radial on the left stick and scalar
on the right, rescaled so the command starts at 0 just outside it. The robot's controller clamps
each axis to +-1 and its walk policy adds its own deadband and filter. The stream runs at the
producer rate while Active; one zero command is sent when the headset pauses, the skeleton is lost
or the producer exits, and nothing is sent while paused (the controller also zeroes the command
after 500 ms without a sample). The axis signs follow the VRS1 thumbstick convention (+x right,
+y forward) and are verified on the headset; `--walk_invert_lateral` / `--walk_invert_yaw` flip
an axis if a setup needs it (`--measure_fps` logs the command).

## Setup

Requirements: Linux x86_64, git, [pixi](https://pixi.sh), network access to GitHub, conda-forge
and PyPI. ROS 2 Jazzy, GMR, `vr_link` and the `igris_c_sdk` wheel are all installed into the
pixi environment; no host ROS installation is used.

```bash
git clone --recurse-submodules <this repo>
cd igris_c_teleop
scripts/setup.sh          # submodules + pixi install + check-env
```

Or step by step:

```bash
git submodule update --init --recursive
pixi install
pixi run check-env        # should print paths under .pixi/envs and `True`
pixi run test             # unit and contract tests
pixi run test-ros         # real rclpy round trip
```

## Run

Start order matters: the inference node exits if it receives no frames within 5 s, and this
producer publishes nothing until a headset skeleton is connected.

1. Start the robot controller / bridge (DDS `lowstate`, `handcmd`, control-mode service).
2. Start the teleop session and put the headset in **Active**. Wait for
   `VR skeleton connected` in the log.
   ```bash
   pixi run teleop -- --dds_namespace igris_c_<unit> --dds_domain_id <id> --ros_domain_id <id> --target robot
   ```
   `teleop` runs the producer and `vr-link-stream` (camera and robot state back to the headset)
   as two supervised processes. `--target` selects the frame source: `robot`, `usb`, `mujoco`,
   `isaac` or `shm`. Extra arguments are passed to the producer.

   `--target` also fixes how the reference frame is published, for `teleop` and `producer` alike:

   | `--target` | Reference frame | Consumer |
   |---|---|---|
   | `robot`, `usb`, `mujoco`, `shm` | ROS 2 `/gmr/teleop/retarget_frame` | inference node |
   | `isaac` | CycloneDDS `<ns>/teleop/retarget_frame`, no ROS 2 publisher | `hit-igris-c-integration` `sim_robot` |

   With `isaac`, `teleop` also pins CycloneDDS to `lo` (unless `$CYCLONEDDS_URI` or `--dds_interface`
   is set), turns the local viewer off and uses the `h264_nvenc` encoder:
   `pixi run teleop -- --dds_namespace igris_c_999 --target isaac`.

   The three identifiers must match the other side, or nothing arrives and no error is shown:

   | Option | Must match | Default |
   |---|---|---|
   | `--dds_namespace` | robot bridge namespace (`<ns>/handcmd`, `<ns>/lowstate`, ...) | none, required |
   | `--dds_domain_id` | robot bridge DDS domain | 0 |
   | `--ros_domain_id` | inference node `domain_id` (`command.yaml`) | 91 |
3. Start the inference node. Step 2 retargets the operator's motion and publishes the whole-body
   reference frame on `/gmr/teleop/retarget_frame`; the inference node subscribes to that frame
   (`MotionReferenceSource`), runs the policy and sends the final joint command to the robot. It is a
   separate repository: see [igris_c_inference_public](https://github.com/robrosinc/igris_c_inference_public)
   for its setup, configuration and run commands.

Teleoperation without vision feedback (pass-through): run the same `teleop` session and check
**NO CAMERA** on the headset's connect page. The headset shows the room in pass-through, and
`vr-link-stream` keeps sending the state records (`Signals`, robot state) without video. The
session has to be `teleop` in this case too: the headset leaves Resuming for Active only when
the producer's resume acknowledgement arrives through `vr-link-stream`.

The producer alone, for development and debugging:

```bash
pixi run producer -- --dds_namespace igris_c_<unit> --dds_domain_id <id> --ros_domain_id <id>   # producer only, nothing sent back to the headset
pixi run producer -- --help                                                                     # all options
```

`producer` runs the retargeting pipeline alone, without `vr-link-stream`. It takes the same
`--target` (default `robot`) and publishes the reference frame by the table above. It cannot
run a live session with `vr_system` v2.2.0 or newer: nothing is sent back to the headset, so the
headset stays in Resuming (HUD `WAITING LINK`) and never becomes Active, with or without
NO CAMERA. Use it to debug the producer or to replay a recording (`--replay_vrs1 <file>.vrs1`).

Useful options: `--no_dds` (ROS output only), `--dds_topic_naming native|ros_compatible` (must
match the robot's transport setting), `--measure_fps` (loop statistics in the log).

Viewer (on by default, `--no_viewer` to disable): `teleop` and `producer` open a MuJoCo window on
this PC. It shows the robot at the published reference pose (after the neck override, arm hold and
convergence ramp) and the GMR IK targets of the same tick, i.e. the scaled human skeleton, as RGB
frames joined by bones. The arm targets have position weight 0 in the GMR config (orientation
only), so they do not sit on the robot's arm links. Display only, 30 Hz (`--viewer_fps`). On a host
without a display (`DISPLAY` / `WAYLAND_DISPLAY` not set, e.g. a headless NUC over plain ssh) the
viewer is skipped with a warning and the producer runs as before. The window is rendered inside the
producer process:

- Keep the window open during a session. Closing it stalls the 100 Hz loop once for tens of ms
  (17 to 52 ms measured); the producer then keeps running without the viewer. Close it only while
  the headset is Paused, or leave it until Ctrl+C.
- A lost X connection or a GL driver crash ends the producer. Pass `--no_viewer` over `ssh -X` and
  wherever the display is not a local desktop session.
- A `DISPLAY` that is set but not reachable (stale value in tmux, dropped `ssh -X`) ends the
  producer at startup with MuJoCo's "could not initialize GLFW", before anything is published.
  Pass `--no_viewer`.
- `--measure_fps` prints the viewer cost per update next to the IK time.

The startup log prints the resolved topic names and domain ids. Compare them with what the robot
bridge and the inference node actually use before trusting a silent run.

## Verification checklist

Before the first use on hardware, and after bumping a submodule or the SDK wheel:

1. `pixi run check-env`, `pixi run test`, `pixi run test-ros` pass.
2. With a live headset in **Active**: `ros2 topic hz /gmr/teleop/retarget_frame` reports the
   headset's record rate (`ROS_DOMAIN_ID=91`; about 52 Hz in a 2026-11 capture, 100 Hz with
   `--no_source_timing`), and `--measure_fps` prints the same number as `solve` next to
   `loop 100 Hz`. The inference node reports ready within 5 s of starting and its
   `[Reference timing]` log shows no `STOPPED` while the headset streams. Publishing stops on
   headset disconnect and resumes on reconnect.
3. With torque off: hand command arrives on the bridge at 100 Hz; each controller button sends
   exactly one control-mode request per press.
4. With torque on, safe posture: Pause the headset, move the arms away, Resume. The arms move
   smoothly to the operator's pose with no jump, and the HUD shows the converging indicator
   until the log reports `convergence ramp finished`.
   Lower body, in MuJoCo before the robot: Pause, change the lower-body posture a little (bend
   the knees, turn the waist), Resume. With the default `--converge_joints upper` the waist and
   leg reference steps to the operator's posture at once; the robot must follow without losing
   balance. Increase the posture difference stepwise, and use `--converge_joints all` if the
   step is too much for the policy.
5. Bare hands: cover one hand. That arm stays still while the rest follows; it ramps back
   smoothly after uncovering.
6. Walk velocity, torque off first: with the headset Active, push each thumbstick axis and read
   `cmd_vel` on the bridge (`--measure_fps` prints vx/vy/yaw): left forward gives +vx, left
   left gives +vy, right left gives +yaw rate, release gives 0, Pause sends one 0 and then
   nothing. Only then, in walk mode with torque on and a clear floor, at the default maxima.
7. Viewer (opens by default), on each PC that runs the session, torque off first: with
   `--measure_fps` the log still reports `loop` about 100 Hz and the same `solve` rate and IK time
   as with `--no_viewer`,
   and the viewer cost stays far below the 10 ms tick (dev PC: 0.4 ms avg, below 1 ms max).

Rollback to the unshaped reference: `--no_converge --arm_hold off --no_motion_feedback_dds`
(with a headset build that waits for the resume acknowledgement keep the motion feedback on:
`--no_converge --arm_hold off`).
Rollback of the resume handling to the joints-only ramp, each switch on its own:
`--no_resume_anchor` (heading/position alignment), `--no_converge_root` (root follows the ramp),
`--converge_per_joint` (joints in step).
Rollback of the upper-body-only ramp to the whole-body ramp: `--converge_joints all`.
Rollback of the viewer: `--no_viewer`.
Rollback of the reference timing (one frame per headset record, headset-clock velocities,
`session_id` / `source_*` in `record_meta`): `--no_source_timing`.

## Submodules

- `thirdparty/gmr`: fork of [GMR (General Motion Retargeting)](https://github.com/YanjieZe/GMR)
  with the IGRIS-C robot model and VR uplink conventions added.
- `thirdparty/vr_link`: headset protocol, TCP receiver and the `vr-link-stream` downlink sender.

## License

`thirdparty/gmr` is a modified version of [GMR](https://github.com/YanjieZe/GMR),
Copyright 2025 Yanjie Ze, released under the **MIT License**. The original license text is kept
in `thirdparty/gmr/LICENSE` and applies to our modifications of that code as well. Robot assets
bundled with GMR (for example `assets/unitree_g1`, `assets/pal_talos`) and vendored utilities
carry their own license files next to them.
