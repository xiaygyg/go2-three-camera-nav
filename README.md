# Go2 Three-Camera Navigation

MuJoCo navigation experiments for a **Unitree Go2** with **front / left / right depth cameras**. Includes a trained high-level checkpoint, full visual meshes, a 40 m sprint approach, a dense forest, and continuous multi-goal navigation.

[中文快速开始](docs/QUICKSTART_ZH.md) · [Inputs and outputs](docs/INPUT_OUTPUT.md) · [Third-party sources](THIRD_PARTY.md)

## 前言：从 NavRL 到机器狗自主导航

本项目基于 [NavRL 开源项目](https://github.com/Zhefan-Xu/NavRL) 的网络结构与局部导航思路，在 MuJoCo 中适配并训练机器狗的上层导航策略。上层输出三维速度命令 **`[vx, vy, wz]`**，分别对应机体坐标系下的前进速度、横移速度和偏航角速度，由底层运动控制策略执行。感谢 [@松鼠山居民](https://space.bilibili.com/26397835) 分享并开源 NavRL，也感谢相关开源项目提供的模型、代码与控制策略。本仓库公开的 **Go2 版本使用 SEA-Nav 提供的冻结底层策略**，上层导航策略在 Go2 关节动力学仿真中继续训练；底层权重来源详见 [THIRD_PARTY.md](THIRD_PARTY.md)。

环境感知采用覆盖全周方向的 **36 个扫描槽位（每 10° 一个方向）**。这里需要区分“全向输入格式”和“全向有效观测”：当前三相机版本从朝前、朝左、朝右的深度图构建局部扫描，同时输入各方向的可见性；后方和相机接缝盲区保持未知，并不拥有完整的 360° 障碍物信息。实际网络扫描张量为 `1 × 36 × 4`，也不是仅输入 36 个距离值；完整接口见 [输入输出说明](docs/INPUT_OUTPUT.md)。导航策略不接收全局障碍物地图。

后续实机部署的关键之一，是稳定、准确地获取这些方向上的障碍物距离和可见性。当前重点探索 **RealSense 深度图 → 三维点云 → 机体坐标系融合 → 按方向聚合扫描** 的方案。仿真中已接入三台名义 D435i 深度相机，实机仍需完成内外参标定、姿态补偿、时间同步及噪声和盲区处理。该路线有望以较低的传感器成本完成局部感知，并为后续引入 RGB 语义信息保留接口；目前 RGB 尚未作为策略输入，也尚未完成实机导航验证。欢迎在 Issues 中交流更稳健的深度转扫描、盲区处理和多相机融合方法！

## 训练环境、配置与实际记录

以下为发布权重对应的 **Go2 适配训练轮次**，不包含早期导航权重的累计训练、底层策略训练、依赖安装、评估或录屏耗时。仅列硬件和软件配置，不包含机器地址及账号。

| 项目 | 配置 / 实测记录 |
|---|---|
| 已验证运行环境 | Ubuntu 24.04.4 LTS / ARM64，NVIDIA GB10（DGX Spark） |
| 软件版本 | Python 3.12.14，PyTorch 2.12.0+cu130，CUDA runtime 13.0，MuJoCo 3.11.0，EGL 无窗口渲染 |
| 动力学与控制 | Go2 完整 12 关节动力学；物理 200 Hz / 底层 50 Hz / 上层导航 10 Hz |
| 感知配置 | 前 / 左 / 右三相机，yaw 为 0° / +90° / −90°；每台 128 × 80，名义视场 87° × 58°；深度上限 8 m，扫描范围 4 m；安装参数待实机标定 |
| 训练方式 | 从已有实验导航权重继续训练；底层冻结；200 次合成教师预热，随后 PPO + 训练期 BC 正则；安全过滤和持续方向脱困参与训练 |
| 采样与更新 | 8 个环境，seed 2222；每环境每次 rollout 64 步，256 次更新，合计 131,072 次环境交互；每次导航交互推进多个物理步，不等于 131,072 个 200 Hz 物理积分步 |
| PPO 参数 | 学习率 1e-4，4 个优化 epoch，minibatch 256，clip 0.2，γ=0.99，GAE λ=0.95；BC 权重 0.15 |
| 任务设置 | 森林局部导航；目标采样距离 3–35 m；单回合上限 900 次导航交互，即 90 s 仿真时间 |
| 输出命令范围 | vx ∈ [−0.5, 1.7] m/s，vy ∈ [−0.8, 0.8] m/s，wz ∈ [−1, 1] rad/s |
| 本轮训练耗时 | **1,651.93 s，约 27 分 32 秒**（训练器计时，包含预热、环境初始化与 PPO） |
| 训练结束状态 | 199 个已完成回合，训练回合到达率约 55.3%；已完成回合碰撞 / 跌倒率均为 0；这些是训练统计，不是独立测试成功率 |

该轮训练从已有导航权重出发，且使用教师预热、BC、安全层及显式脱困，因此不能把 27 分钟理解为“从零学会完整导航”的时间，也不能推广到其他硬件或训练预算。此前实验中提及的约 14–15 分钟并非此 Go2 发布权重的训练时长。训练器日志中的 `physics_transitions` 字段实际计数的是环境交互次数。

| 已完成的回归测试（每条路线仅测试首个目标，90 s 上限） | 到达 | 超时 | 碰撞 / 跌倒 |
|---|---:|---:|---:|
| 已知 6 任务，启用显式脱困 | 5 / 6 | 1 | 0 / 0 |
| 已知 6 任务，关闭显式脱困 | 5 / 6 | 1 | 0 / 0 |
| 带噪声 12 任务，启用显式脱困 | 9 / 12 | 3 | 0 / 0 |
| 带噪声 12 任务，关闭显式脱困 | 6 / 12 | 6 | 0 / 0 |

关闭显式脱困时仍保留底层控制和安全过滤，不代表无任何辅助的裸网络。上述测试来自原适配实验的已使用回归集、单个训练 seed，不能作为独立泛化或统计显著性结论。发布演示另测一条连续任务：**40 m 直道后进入森林，5 / 5 目标到达，零碰撞、零跌倒、零机器人重置**；它与上表的多任务回归口径不同，速度与视频记录见下文。

下文的公开训练命令是**新训练的起点**（默认 800 次预热、horizon 600、BC 0.1），不是上述历史续训的逐项重放；历史初始化权重没有随本仓库发布，因此不承诺从头运行得到相同权重或耗时。可通过 `--warmup 200 --horizon 900 --bc-weight 0.15` 对齐本轮相应超参数，使用 `--resume` 时需提供与当前 Go2 三相机接口兼容的权重。

![Sprint-to-forest demo](media/demo.gif)

Full video and measured speed results are provided in [Releases](https://github.com/xiaygyg/go2-three-camera-nav/releases).

## What runs

- Articulated Go2 dynamics: 12 joints, 200 Hz physics, frozen 50 Hz locomotion controller.
- Three rendered depth images: forward, left and right; rear and camera seams remain unknown.
- NavRL-inspired, locally trained 10 Hz navigation policy plus depth-only safety and persistent escape supervision.
- A **40 m straight approach** before entering a **66 × 46 m forest** with 260 trees, 30 rocks, and five sequential goals. The sprint scene adds four trees near the entrance/route.
- Continuous handoff and goal changes, with no robot or locomotion-controller resets.
- H.264 recording with depth images, local scan / visibility, target direction, commanded and measured body velocity, and a display-only scene overview.

This is an independent MuJoCo quadruped adaptation. It is not the original aerial NavRL task or the official Isaac Gym SEA-Nav benchmark. SEA-Nav supplies the frozen locomotion modules; the navigation policy here has its own observation and training interface.

## Recorded release demo

One predeclared seeded route, using the released checkpoint and default recovery:

| Measured result | Value |
|---|---:|
| Straight requested speed | 1.70 m/s |
| Straight mean forward speed after startup | 1.30 m/s |
| Straight peak planar speed | 1.44 m/s |
| Steady straight vx tracking RMSE | 0.40 m/s |
| Forest mean planar speed | 0.83 m/s |
| Goals reached | 5 / 5 |
| Collision / fall / robot reset | 0 / 0 / 0 |
| Simulation duration | 182 s |

![Executed vs measured speed](media/speed_tracking.png)

This is one continuous demonstration, not a multi-seed success-rate claim. The GIF is a 15-second excerpt from separate moments of the uncut recording. The tracking bias is retained and reported rather than treating a 1.7 m/s request as achieved speed. Numeric results are in `media/demo_metrics.json`; the full video and telemetry CSV are release assets.

## Quick start (Linux)

Python 3.10+ and an OpenGL rendering backend are required. Linux with NVIDIA EGL is the tested headless configuration. The high-level policy can run on CPU; PyTorch CUDA is optional for training.

```bash
git clone https://github.com/xiaygyg/go2-three-camera-nav.git
cd go2-three-camera-nav
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python scripts/download_assets.py
pytest -q
```

Public meshes, locomotion models and the optional NavRL initialization checkpoint are fetched at pinned revisions and checked against SHA-256. They are not silently substituted if verification fails. Their upstream terms remain applicable.

### Record the sprint-to-forest demo

```bash
bash scripts/run.sh scripts/run_go2_patrol.py \
  --video --approach-length 40 --sprint-mode command --sprint-speed 1.7 \
  --seconds-per-goal 120 --name my_sprint_forest
```

The straight segment requests **1.7 m/s**, the supported upper bound of this navigation configuration. This is a tracking test, **not a claim about Go2 hardware maximum speed**. On crossing the forest entrance, the navigation policy and its depth-only supervisor take over. The three cameras remain active throughout. Inspect measured speed, tracking error, collisions and completed goals rather than the desired speed alone.

Add `--viewer` for an interactive MuJoCo window on a desktop. On a desktop system without EGL, set `MUJOCO_GL=glfw`. Output names must be new; recordings are never silently overwritten.

```bash
# No straight approach: continuous random-goal forest navigation.
bash scripts/run.sh scripts/run_go2_patrol.py \
  --approach-length 0 --random-goals 5 --seed 2026 \
  --video --seconds-per-goal 120 --name random_patrol
```

Targets are supplied as coordinates and projected into the robot frame. This demo does not perform semantic object detection. Procedural scene construction reserves traversable corridors; the policy does not receive these corridors or a global obstacle map.

### Train a new high-level policy

```bash
# Fresh output directory required. Lower locomotion models stay frozen.
bash scripts/run.sh scripts/train_go2_navigation.py \
  --sensor three_camera --scenario forest --envs 8 \
  --warmup 800 --steps 131072 --seed 2222 \
  --output runs/go2_camera_training

# Short installation check; not a useful trained policy.
bash scripts/run.sh scripts/train_go2_navigation.py \
  --envs 2 --warmup 1 --steps 128 --output runs/install_check
```

Training uses a local synthetic teacher warm-up followed by PPO in articulated physics, with a training-only behavior-cloning regularizer. Safety and explicit recovery are part of the training environment. Increasing `--steps` increases the budget; it does not guarantee success. Use `--checkpoint runs/go2_camera_training/final.pt` with the demo to test your own policy.

### Evaluate fixed tasks

```bash
bash scripts/run.sh scripts/evaluate_go2_navigation.py \
  --checkpoint assets/checkpoints/go2_three_camera.pt \
  --plan configs/evaluation_routes.json --seconds 90 --name fixed_eval
```

This evaluator scores the **first goal of each listed route**, retaining failures. It is not an evaluation of every later waypoint. Use the patrol demo for a complete continuous mission. Recovery is enabled by default; `--no-recovery` is an explicit ablation. Experimental lateral-first escape modes are opt-in and are not claimed to outperform the default.

## Files

| Path | Purpose |
|---|---|
| `go2_nav/` | Go2 dynamics, high-level policy, frozen lower-policy interface, recovery and recording |
| `go2_nav/common/` | Camera processing, scan safety, scenes, training teacher and shared numerical utilities |
| `configs/go2_three_d435i.json` | Explicit uncalibrated simulation camera rig |
| `assets/checkpoints/go2_three_camera.pt` | Portable navigation weights; no machine-specific metadata |
| `assets/sources.json` | Pinned download URLs, sizes and hashes |
| `scripts/` | Download, training, evaluation and demo entry points |
| `docs/INPUT_OUTPUT.md` | Tensor shapes, coordinates, units and control limits |
| `media/` | Short demonstration preview and public measured results |

The camera configuration is a simulation assumption, not a calibrated hardware installation. No hardware performance or deployment safety is established by this repository. Dynamics-based rewards, collision checks and procedural task sampling use simulator state; camera-mode navigation does not use geometric obstacle ground truth for sensing.

## License and attribution

Original adaptation code is MIT licensed. NavRL-derived components retain [NavRL's MIT notice](docs/NavRL-LICENSE.txt). MuJoCo Menagerie assets and SEA-Nav weights remain subject to their own upstream terms; see [THIRD_PARTY.md](THIRD_PARTY.md). The source repository does not bundle those third-party runtime downloads.
