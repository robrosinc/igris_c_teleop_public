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
   ├─ shaping       convergence ramp, arm hold (see below)
   └─ outputs
        ├─ ROS 2  reference motion frame  ──>  igris_c_inference_node  ──>  robot (lowcmd)
        ├─ DDS    hand command            ──>  robot
        ├─ DDS    control-mode request    ──>  robot   (controller buttons)
        └─ DDS    motion feedback         ──>  headset HUD (via vr-link-stream)

vr-link-stream (vr_link)   robot camera + robot state  ──>  headset   (started by `pixi run teleop`)
```

Source layout follows the pipeline (`source/igris_c_teleop/`): `uplink` (headset input),
`retarget`, `shaping`, `ros` / `dds` (outputs), `app` (entry points).

## Reference shaping

The policy applies the reference without any rate limit of its own, so these stages are the
only thing between the operator and a joint jump. All are enabled by default.

- **Convergence ramp** (`--no_converge` to disable). When teleoperation starts or resumes, or
  the robot enters low-level control, the reference is blended from the robot's measured joint
  pose to the operator's pose instead of jumping. The ramp is rate-capped (default 0.5 rad/s) and
  ends as soon as every joint is close to the live reference.
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

   The three identifiers must match the other side, or nothing arrives and no error is shown:

   | Option | Must match | Default |
   |---|---|---|
   | `--dds_namespace` | robot bridge namespace (`<ns>/handcmd`, `<ns>/lowstate`, ...) | none, required |
   | `--dds_domain_id` | robot bridge DDS domain | 0 |
   | `--ros_domain_id` | inference node `domain_id` (`command.yaml`) | 91 |
3. Start the inference node:
   ```bash
   ros2 run igris_c_inference_public igris_c_inference_node --config <policy>/params.yaml
   ```

Other ways to run:

```bash
pixi run producer -- --dds_namespace igris_c_<unit> --dds_domain_id <id> --ros_domain_id <id>   # producer only, no video downlink
pixi run replay -- <file>.vrs1 --ros_domain_id <id>                                             # ROS only, replay a recording without a headset
pixi run producer -- --help                                                                     # all options
```

Useful options: `--no_dds` (ROS output only), `--dds_topic_naming native|ros_compatible` (must
match the robot's transport setting), `--measure_fps` (loop statistics in the log).

The startup log prints the resolved topic names and domain ids. Compare them with what the robot
bridge and the inference node actually use before trusting a silent run.

## Verification checklist

Before the first use on hardware, and after bumping a submodule or the SDK wheel:

1. `pixi run check-env`, `pixi run test`, `pixi run test-ros` pass.
2. `pixi run replay -- <file>.vrs1`: `ros2 topic hz /gmr/teleop/retarget_frame` reports about 100 Hz
   (`ROS_DOMAIN_ID=91`); the inference node attached to the replay reports ready within 5 s.
3. Same with a live headset. Publishing stops on disconnect and resumes on reconnect.
4. With torque off: hand command arrives on the bridge at 100 Hz; each controller button sends
   exactly one control-mode request per press.
5. With torque on, safe posture: Pause the headset, move the arms away, Resume. The robot moves
   smoothly to the operator's pose with no jump, and the HUD shows the converging indicator
   until the log reports `convergence ramp finished`.
6. Bare hands: cover one hand. That arm stays still while the rest follows; it ramps back
   smoothly after uncovering.

Rollback to the unshaped reference: `--no_converge --arm_hold off --no_motion_feedback_dds`.

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
