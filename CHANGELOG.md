# Changelog

Release history of the IGRIS-C VR teleoperation set: `igris_c_teleop` (this repository), the `vr_system` headset app (APK attached to each [GitHub Release](https://github.com/robrosinc/igris_c_teleop_public/releases)) and the `vr_link` protocol both sides share. Setup and operation are described in the [README](README.md); this file records only what changed between versions.

Versions are git tags. `igris_c_teleop` and `vr_system` carry the same tag for one release, and the APK and the producer must be used at the same version. The package version in `pyproject.toml` is not tied to the tags.

## [2.2.0] - 2026-11-27

Thumbstick walking, a resume handshake between the headset and the producer, and reference frames timed by the headset clock.

### Compatible set

| Component | Version |
| --- | --- |
| `igris_c_teleop` | tag `v2.2.0` (`748e882`) |
| `vr_system` APK | `vr_system_v2.2.0.apk` (tag `v2.2.0`, `8ec4cb6`) |
| `vr_link` (submodule) | `341fce0` in `igris_c_teleop`, `47722a6` in `vr_system`. The wire schema is the same: the one commit between them only exposes the record timestamp in the Python receiver |
| `gmr` (submodule) | `ef305fc` |
| `igris_c_sdk` wheel | `v2.1.1`, unchanged |

### igris_c_teleop

#### Added

- **Walk velocity from the thumbsticks.** While the headset is Active the producer publishes `<ns>/cmd_vel` (`TwistStamped`) on DDS: left stick forward/back is `vx`, left stick left/right is `vy`, right stick left/right is the yaw rate. Full deflection gives 0.5 m/s and 0.8 rad/s, the deadzone is 0.1. The robot uses the command in walk mode and ignores it in low-level joint control. One zero command is sent on pause, skeleton loss and exit. Options: `--no_walk_dds`, `--walk_max_linear`, `--walk_max_yaw_rate`, `--walk_deadzone`, `--walk_invert_lateral`, `--walk_invert_yaw`, `--cmd_vel_dds_topic`.
- **Resume acknowledgement.** When the convergence ramp of a Start or Resume has finished, the producer reports that request's `resume_count` back to the headset on `teleop/motion_feedback`. The v2.2.0 headset waits for it before it becomes Active.
- **Resume anchor** (`--no_resume_anchor` disables). A turn or a walk the operator made while paused is not passed to the robot: on every Start/Resume the skeleton is rotated about the vertical axis and shifted horizontally once, so that it continues with the heading and foot position of the last retargeted skeleton.
- **Reference timing by the headset clock** (`--no_source_timing` restores the previous behaviour). Each headset record is retargeted and published once, velocities are differenced over the headset timestamps, and `record_meta` carries `session_id`, `source_timestamp_ns` and `source_session_id`.
- **Reference viewer** (on by default, `--no_viewer` disables). A MuJoCo window on the teleop PC shows the robot at the published reference pose together with the GMR IK targets. Display only, 30 Hz (`--viewer_fps`). It is skipped with a warning on a host without a display.
- `--converge_joints {upper,all}`, `--converge_per_joint` and `--no_converge_root` select what the convergence ramp shapes (see Changed).

#### Changed

- **The convergence ramp shapes the upper body only by default**: shoulders, elbows, wrists and neck, 16 of the 31 joints. The waist and leg reference and the root pose follow the operator from the first frame. `--converge_joints all` ramps every joint as before; the root pose then follows the ramp as well (`--no_converge_root` leaves the root live).
- The ramped joints move in step: one blend sized by the largest gap, so that all joints arrive together. `--converge_per_joint` restores the previous fixed-duration blend with each joint capped on its own.
- The default ramp rate cap is 0.6 rad/s (was 0.5, `--converge_max_rad_s`).
- A ramp that finds no fresh `LowState` (`--lowstate_max_age`, 0.5 s) waits, holding the last published reference and hand command, and starts when one arrives. Before, the ramp was skipped.
- `/gmr/teleop/retarget_frame` is published at the headset's record rate (about 52 Hz measured) instead of the 100 Hz loop rate. The DDS outputs keep 100 Hz.

#### Fixed

- Reference velocities are no longer differentiated across the switch to the robot's measured joints at a ramp start. Velocity estimation is reset at each ramp start, and velocities are zero for the following `--zero_velocity_frames` frames (default 10).

### vr_system (APK)

#### Changed

- **Resuming to Active follows the producer.** Every uplink record carries the id of the current Start/Resume request (`resume_count`). The app streams the live pose while Resuming and becomes Active when the producer echoes that id, that is, when the robot has finished converging.
- HUD: `RESUMING` while the app waits for the producer, `WAITING LINK` when the PC side does not answer for more than 1 s, and the `GO` flash on becoming Active.
- Controller inputs stay at their paused values during Resuming and go live at Active.

#### Removed

- The headset-side resume blend: the app no longer blends the pose from the paused one to the live one. The producer's convergence ramp is the only shaping of a resume.

### vr_link

- VRS1 `VRState.resume_count` and VRL2 `Signals.converged_resume_count` added (`47722a6`).
- DDS `teleop/motion_feedback` uses layout `vr_link.motion_feedback.v2`, which adds the `converged_resume_count` button. `v1` messages are still decoded.
- The Python receiver exposes the record timestamp as `VRInputState.source_timestamp_ns` (`341fce0`).
- The served stereo calibration file is `data/calibration/igris_c_eyes_stereo.yaml` (was `igris_c_eyes_stereo_unreal.yaml`). `vr-link-calibrate-cams` writes it by default, and `stream_dds.yaml` and `stream_usb.yaml` read it (`2a629ec`).

### gmr

- The human scale ratio is `human_height_assumption / actual_human_height` (was the inverse) and the ground offset is 0.05 m (was 0.1 m). The retargeted pose for the same operator input differs from v2.1.3 (`--actual_human_height`, default 1.65).

### Interface changes

- **VRS1 / VRL2**: `VRState.resume_count` and `Signals.converged_resume_count` added.
- **DDS `<ns>/teleop/motion_feedback`**: frame id `vr_link.motion_feedback.v2`, buttons `[converging, arm_hold_left, arm_hold_right, converged_resume_count]`.
- **DDS `<ns>/cmd_vel`** (`TwistStamped`): new topic.
- **ROS 2 `/gmr/teleop/retarget_frame`**: `record_meta` gains `session_id`, `source_timestamp_ns` and `source_session_id`; the publish rate is the headset's record rate.
- The other topics, their names and QoS are unchanged.

### Upgrade notes

- Install `vr_system_v2.2.0.apk` and check out `v2.2.0` together. A v2.2.0 headset stays in Resuming against an older producer, which never sends the acknowledgement.
- **`pixi run producer` alone can no longer run a live session.** The acknowledgement reaches the headset through `vr-link-stream` and the motion feedback. Without them the headset stays in Resuming (HUD `WAITING LINK`): this applies to `pixi run producer`, `--no_stream`, `--no_dds` and `--no_motion_feedback_dds`. For teleoperation without vision feedback, run `pixi run teleop` and check **NO CAMERA** on the headset (pass-through view). The README at this tag still describes `producer` with NO CAMERA; that combination does not work with the v2.2.0 APK.
- The inference node now receives reference frames at the headset's record rate and is expected to place them by `source_timestamp_ns`. `--no_source_timing` restores 100 Hz frames with host-clock velocities.
- With the default upper-body ramp, a lower-body posture change made while paused reaches the policy as a step on Resume. Check it in simulation first; `--converge_joints all` ramps every joint.
- Pass `--no_viewer` over `ssh -X` and wherever the display is not a local desktop session: a lost X connection ends the producer. Keep the viewer window open during a session.
- The stream configs no longer read `data/calibration/igris_c_eyes_stereo_unreal.yaml`. Run `vr-link-calibrate-cams` again, or rename the existing file to `igris_c_eyes_stereo.yaml`.

The DDS output moves a real robot and switches its control mode. Run the [Verification checklist](https://github.com/robrosinc/igris_c_teleop_public/blob/v2.2.0/README.md#verification-checklist) with torque off before connecting hardware; the rollback switches of each change are listed there.

## [2.1.3] - 2026-10-30

Compatible set: `igris_c_teleop` `aa71485`, APK `vr_system-v2.1.3.apk` (`d4d4cd9`), `vr_link` `0ff9a9f`, `gmr` `1281a40`, `igris_c_sdk` wheel `v2.1.1`.

### igris_c_teleop

- Added: ROS 2 topic `/gmr/teleop/active` (`std_msgs/Bool`, TRANSIENT_LOCAL depth 1). It is true only while the headset is Active and false on shutdown.
- Changed: submodule URLs in `.gitmodules` are relative paths. README rewritten.

### vr_system (APK)

- Changed: camera extrinsics. The eye-camera offset is a full transform in the neck-pitch frame instead of a vector, which improves the hand-target overlay.
- Changed: the controller-held check (grip-to-wrist distance gate) is disabled. During fast controller motion it dropped the tracked flag and engaged the arm hold.

### Interface changes

- ROS 2 `/gmr/teleop/active` added.

## [2.1.2] - 2026-10-21

Compatible set: `igris_c_teleop` `aec75aa`, APK `vr_system_v2.1.2.apk` (`c4a747d`), `vr_link` `0ff9a9f`, `gmr` `1281a40`, `igris_c_sdk` wheel `v2.1.1`.

### igris_c_teleop

- Changed: convergence ramp. The default blend is 1.0 s (was 2.0 s) and the ramp ends as soon as every joint is within tolerance of the live reference.
- Changed: with a controller the hand command is rescaled to full grip (1.0). Hand, grip and hold are decided per side.
- Deprecated: the `converge_progress` axis of `teleop/motion_feedback` is sent as 0.

### vr_system (APK)

- Changed: the HUD shows the converging flag only. The progress percentage is removed.

### vr_link

- The hand-target overlay takes its body reference from the inference node's DDS `lowcmd` instead of `gmr/refstate` (packaged stream configs).

### Interface changes

- `converge_progress` is deprecated; the message layout is kept.

## [2.1.1] - 2026-10-20

Compatible set: `igris_c_teleop` `fb3aedc`, APK `vr_system_v2.1.1.apk` (`59f91cc`), `vr_link` `7fa80b6`, `gmr` `1281a40`, `igris_c_sdk` wheel `v2.1.1`.

### igris_c_teleop

- Changed: `vr_link` pin only, no source change.

### vr_system (APK)

- Added: numeric countdown on Start/Resume. The HMD pose is re-aligned right before resuming.

### vr_link

- Fixed: encoder threading delay removed. Left/right swap of the side-by-side video corrected (the swap applies to USB only).

## [2.1.0] - 2026-10-19

Compatible set: `igris_c_teleop` `daa8a6c`, APK `vr_system_v2.1.0.apk` (`9a634e5`), `vr_link` `de67002`, `gmr` `1281a40`, `igris_c_sdk` wheel `v2.1.1`.

### igris_c_teleop

- Added: head camera stream option in the `teleop` session.

### vr_system (APK)

- Added: the head camera (D435 RGB) is received and decoded as a second video stream. Lower-latency video path.

### vr_link

- Added: `VideoStream {EyesStereo, HeadRgb}` and `stream_id`, with an independent H.264 sequence per stream (schema 2.2).

### Interface changes

- VRS1 / VRL2 schema 2.2 (second video stream).

## [2.0.0] - 2026-10-19

First tagged baseline of the restructured package. Tag only: no GitHub Release and no APK attached.

Compatible set: `igris_c_teleop` `f5ba9b3`, `vr_system` `396d461`, `vr_link` `edacc32`, `gmr` `1281a40`, `igris_c_sdk` wheel `v2.1.1`.

### igris_c_teleop

- Repository renamed from `igris_c_gmr` and reorganised into `uplink`, `retarget`, `shaping`, `ros`, `dds` and `app`.
- Reference shaping added: convergence ramp and arm hold, with `teleop/motion_feedback` for the headset HUD.
- `vr-link-stream` integrated into the `teleop` session.
- `igris_c_sdk` is installed from the public release wheel.

### vr_system (APK)

- Pass-through view mode. Left/right image swap in the vision feedback fixed.
- Controller tracking state uplinked, hand overlay toggle, motion-feedback status UI, controller key bindings.

### vr_link

- Controller tracked flag in the uplink, teleop state in `VRInputState`, `teleop/motion_feedback` reception.

[2.2.0]: https://github.com/robrosinc/igris_c_teleop_public/compare/v2.1.3...v2.2.0
[2.1.3]: https://github.com/robrosinc/igris_c_teleop_public/compare/v2.1.2...v2.1.3
[2.1.2]: https://github.com/robrosinc/igris_c_teleop_public/compare/v2.1.1...v2.1.2
[2.1.1]: https://github.com/robrosinc/igris_c_teleop_public/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/robrosinc/igris_c_teleop_public/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/robrosinc/igris_c_teleop_public/tree/v2.0.0
