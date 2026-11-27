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

Each [release](https://github.com/robrosinc/igris_c_teleop_public/releases) pairs a tag of this
repository with the matching `vr_system` headset app (APK attached to the release). Use both at
the same version. [CHANGELOG.md](CHANGELOG.md) lists what changed in each version.

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
   as two supervised processes. `--target` names what the session drives: `robot`, `usb`,
   `mujoco`, `isaac` or `shm`. Extra arguments are passed to the producer.

   The three identifiers must match the other side, or nothing arrives and no error is shown:

   | Option | Must match | Default |
   |---|---|---|
   | `--dds_namespace` | robot bridge namespace (`<ns>/handcmd`, `<ns>/lowstate`, ...) | none, required |
   | `--dds_domain_id` | robot bridge DDS domain | 0 |
   | `--ros_domain_id` | inference node `domain_id` (`command.yaml`) | 91 |
3. Start the inference node. It subscribes to the reference frame on
   `/gmr/teleop/retarget_frame`, runs the policy and sends the joint command to the robot. It is
   a separate repository: see
   [igris_c_inference_public](https://github.com/robrosinc/igris_c_inference_public).

The startup log prints the resolved topic names and domain ids. Compare them with what the robot
bridge and the inference node actually use before trusting a silent run.

Variants:

- **Without vision feedback (pass-through).** Run the same `teleop` session and check
  **NO CAMERA** on the headset's connect page.
- **Simulator.** `--target mujoco` or `--target isaac`. With `isaac` the reference frame goes
  out over CycloneDDS (`<ns>/teleop/retarget_frame`) instead of ROS 2.
- **Producer alone.** `pixi run producer -- --dds_namespace igris_c_<unit>` runs without
  `vr-link-stream`, for debugging and for replaying a recording (`--replay_vrs1 <file>.vrs1`).
  It cannot run a live session with `vr_system` v2.2.0 or newer: the headset stays in Resuming.
- **Viewer.** A MuJoCo window shows the published reference pose. It is skipped on a host
  without a display; pass `--no_viewer` over `ssh -X`.

`pixi run producer -- --help` lists all options.

## Controls

Controller input is read only while the headset is Active. Right A belongs to the headset itself
(Start / Pause / Resume).

| Input | Action |
|---|---|
| Right B | control mode: low-level joint control (teleoperation) |
| Right thumbstick click | control mode: joint position hold |
| Left thumbstick click | control mode: motion preset HOME |
| Left thumbstick | walk forward / backward, step left / right |
| Right thumbstick left / right | turn left / right |

The thumbsticks publish a walk velocity (`cmd_vel`, up to 0.5 m/s and 0.8 rad/s). The robot uses
it in walk mode and ignores it in low-level joint control. `--no_walk_dds` disables it.

## Reference shaping

The policy applies the reference without a rate limit of its own, so the producer shapes it
before publishing. All of these are on by default:

- **Convergence ramp** (`--no_converge`). On Start / Resume and when the robot enters low-level
  control, the upper-body reference ramps from the robot's measured pose to the operator's pose
  instead of jumping. The waist and leg reference is not ramped; `--converge_joints all` ramps
  every joint.
- **Resume anchor** (`--no_resume_anchor`). A turn or a walk the operator made while paused is
  not passed to the robot.
- **Arm hold** (`--arm_hold off`). An arm whose bare hand leaves the headset's tracking view is
  frozen, and ramps back when the hand is tracked again.
- **Motion feedback** (`--no_motion_feedback_dds`). The converging and arm-hold states are shown
  on the headset HUD. The headset also needs this feedback to become Active, so keep it on.

The reference frame is published once per headset record (about 52 Hz) and timed by the headset
clock; the DDS outputs run at 100 Hz. `--no_source_timing` restores one frame per 100 Hz tick.

## Verification checklist

Before the first use on hardware, and after bumping a submodule or the SDK wheel:

1. `pixi run check-env`, `pixi run test`, `pixi run test-ros` pass.
2. Live headset in **Active**: `ros2 topic hz /gmr/teleop/retarget_frame` (`ROS_DOMAIN_ID=91`)
   reports the headset's record rate, about 52 Hz. The inference node reports ready within 5 s.
   Publishing stops on headset disconnect and resumes on reconnect.
3. Torque off: the hand command arrives on the bridge at 100 Hz; each controller button sends
   exactly one control-mode request per press.
4. Torque on, safe posture: Pause the headset, move the arms away, Resume. The arms move
   smoothly to the operator's pose with no jump. Check a lower-body posture change across
   Pause / Resume in simulation before the robot: that reference is not ramped by default.
5. Bare hands: cover one hand. That arm stays still while the rest follows; it ramps back
   smoothly after uncovering.
6. Walk velocity, torque off first: push each thumbstick axis and read `cmd_vel` on the bridge:
   left forward gives +vx, left left gives +vy, right left gives +yaw rate, release gives 0.
   Only then, in walk mode with torque on and a clear floor.
7. Viewer: with `--measure_fps` the log still reports `loop` about 100 Hz with the window open.

Rollback to the unshaped reference: `--no_converge --arm_hold off`. Single switches:
`--converge_joints all` (whole-body ramp), `--no_resume_anchor`, `--no_source_timing`,
`--no_viewer`.

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
