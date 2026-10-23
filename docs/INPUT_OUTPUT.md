# Inputs and outputs / 输入输出

This document describes the **three-camera MuJoCo Go2 adaptation**, not the separate 550-dimensional official SEA-Nav navigation interface.

## High-level observation (10 Hz)

The actor receives three float32 tensors. Shapes below exclude the batch dimension.

| Tensor | Shape | Meaning |
|---|---|---|
| State | `(15,)` | Local goal, body motion, previous command and recovery state |
| Depth-derived scan | `(1,36,4)` | 36 angular bins, each with visibility and three replicated distance features |
| Dynamic tracks | `(1,5,10)` | Reserved for tracked obstacles; **all zeros** in the released three-camera configuration |

State indices are zero-based:

| Indices | Input | Units / processing |
|---|---|---|
| `0:2` | Unit direction to the current target | Goal expressed in horizontal body-yaw frame |
| `2` | Progress-stall timer | `min(1, stall_seconds / 3)` |
| `3` | Target distance | `min(distance_m, 20)`; actor additionally divides this by 10 |
| `4` | Explicit escape supervisor active | 0 or 1 |
| `5:8` | Body linear velocity x/y/z | m/s; actor additionally normalizes x by 1.7 |
| `8:11` | Body angular velocity x/y/z | rad/s multiplied by 0.25 |
| `11:14` | Previous executed vx/vy/wz | Divided by `[1.7, 0.8, 1.0]` |
| `14` | Body height offset | `2 * (height_m - 0.32)` |

The network's `state_scale` performs its additional internal normalization; do not apply that normalization twice in an external adapter. The two recovery features are provided by the runtime supervisor and must be preserved for the released checkpoint.

### Depth images → scan

- Nominal D435i-like front/left/right mounts at yaw 0°/+90°/−90°.
- Rendered image size 128×80 per camera; nominal horizontal/vertical FOV 87°/58°.
- Depth measured in meters; render range 8 m, navigation range 4 m.
- Camera intrinsics and optical-to-body transforms project depth into body-frame points. Height filtering and a conservative 0.55 m footprint produce local obstacle clearance.
- Angular bin `i` corresponds to `i * 10°` around the body. The 36-bin array is a full-circle **indexing format**, not a guarantee of full-circle sensing.
- Scan column 0 is the visibility fraction. Columns 1–3 repeat `4 − clipped_clearance_m`. Unknown bins retain zero visibility; the scan alone must not be interpreted as free space without the visibility channel.
- Rear sectors, seams, invalid depth and camera dropout remain unobserved. The runtime guard forbids translation into directions lacking sufficient valid coverage.

Only rendered camera depth is used for obstacle sensing in this mode. Simulator geometry is used for training rewards, collision termination, task placement and the diagnostic overview, not as a replacement for unseen camera data. There is no global-map tensor or semantic goal detector. The displayed target is a supplied coordinate transformed using robot pose.

Code: `go2_nav/environment.py::local_observation`, `go2_nav/common/three_camera_depth.py`, `go2_nav/common/camera_observation.py`, `go2_nav/recovery_env.py::observe`.

## Navigation output → executed command

The actor produces a Beta distribution during training and uses its mean during deterministic evaluation. Three outputs map to body velocities:

| Output | Bounds | Units |
|---|---|---|
| vx | −0.5 … +1.7 | m/s |
| vy | −0.8 … +0.8 | m/s |
| wz | −1.0 … +1.0 | rad/s |

The command may then be changed by persistent escape supervision, depth-based safety filtering, blind-sector gating and slew limits. Per 0.1 s navigation step, command changes are limited to ±0.7 m/s, ±0.4 m/s and ±0.5 rad/s. Navigation output, requested sprint command, shielded command, final executed command and measured velocity are different signals; the demo's CSV keeps those stages separate.

In the default straight approach, a commanded 1.7 m/s tracking test overrides the navigation request. After the entrance, the actor and supervisor control navigation. `--sprint-mode policy` explicitly removes that straight-segment override; it does not force the actor to request maximum speed.

## Frozen locomotion (50 Hz) and physical control (200 Hz)

| Item | Shape / value |
|---|---|
| Lower-policy proprioception | 45: angular velocity 3, projected gravity 3, scaled command 3, joint position offsets 12, joint velocities 12, previous action 12 |
| Lower history | 10×45 = 450 |
| Velocity encoder output | 2 horizontal components |
| Latent encoder output | 16 |
| Body actor input | 2 + 45 + 1 extra yaw-rate feature + 16 = 64 |
| Body actor output | 12 joint actions |
| Joint target | default joint angle + 0.25 × reindexed action |
| PD gains | Kp=30, Kd=0.75 |
| Physics time step | 0.005 s; four physics steps per locomotion step |

MuJoCo joint order is FL/FR/RL/RR; released model order is FR/FL/RR/RL. The permutation is `[3,4,5,0,1,2,9,10,11,6,7,8]`. All lower models remain frozen during high-level training. The measured velocity shown in video is computed from simulated body motion; it is not a command or the lower model's velocity estimate.

## Camera calibration and real hardware

`configs/go2_three_d435i.json` is an explicit **uncalibrated simulation assumption**. Real depth images require calibrated intrinsics/extrinsics, synchronized timestamps, depth validity and filtering, and an independent pose/velocity estimate. Current simulation observations use exact body state; the repository does not supply a production VIO/SLAM system or a robot safety controller.
