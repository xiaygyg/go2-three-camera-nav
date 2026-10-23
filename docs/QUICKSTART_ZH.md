# 中文快速开始

本项目是 Go2 三深度相机 MuJoCo 发布版。高层为本项目训练的导航策略，底层为冻结的 SEA-Nav Go2 步态模型。包含完整mesh、前/左/右深度图、后方与相机接缝盲区、速度跟踪长跑道和密集树林多目标任务。

## 安装

推荐 Linux＋Python 3.10及以上；无桌面运行采用NVIDIA EGL。代码不依赖任何指定机器、用户名或服务器地址。

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

下载器固定公开上游版本并核对SHA256。先阅读THIRD_PARTY.md，模型和mesh的许可不因本项目发布而改变。导航checkpoint已随仓库提供，底层公开模型与mesh单独下载。

## 长跑道＋树林连续演示

```bash
bash scripts/run.sh scripts/run_go2_patrol.py \
 --video --approach-length 40 --sprint-speed 1.7 \
 --seconds-per-goal 120 --name first_demo
```

先在40米直道执行1.7m/s速度指令，验证底层跟踪；到树林入口后切换为高层导航＋安全与脱困控制，依次寻找5个目标。机器人和底层历史不重置。1.7是当前配置的指令上限，不是实机最高速度；请查看报告中的实测速度与误差。直道是明确给定速度的测试，不能称为策略自己选择了这个速度。

`results/first_demo.mp4`为H.264视频；同名JSON、CSV和路线图保存全过程、实测速度、目标事件和失败。失败或超时不会被删除，也不会自动重试直到成功。输出名必须全新。

有桌面时加入`--viewer`；如不用EGL，可先`export MUJOCO_GL=glfw`。窗口中的概览图只是给人看的，不会传给策略。随机目标可使用：

```bash
bash scripts/run.sh scripts/run_go2_patrol.py --approach-length 0 \
 --random-goals 5 --seed 2026 --video --name random_demo
```

## 训练

```bash
bash scripts/run.sh scripts/train_go2_navigation.py \
 --sensor three_camera --scenario forest --envs 8 \
 --warmup 800 --steps 131072 --seed 2222 --output runs/new_policy
```

训练高层导航，底层模型冻结；先进行局部教师预热，再做关节动力学环境中的PPO，并保留训练用BC项、安全层和显式脱困。训练预算不保证导航效果；评估时须使用相同相机与脱困配置。可把`--envs`改成适合显存的值，环境步数可能向完整rollout取整。

快速验证训练入口：

```bash
bash scripts/run.sh scripts/train_go2_navigation.py \
 --envs 2 --warmup 1 --steps 128 --output runs/smoke
```

新模型测试：在演示命令中增加`--checkpoint runs/new_policy/final.pt`。发布checkpoint的默认脱困模式为`legacy`；横移优先候选保留为可选实验，不宣称更优。

## 任务评估

```bash
bash scripts/run.sh scripts/evaluate_go2_navigation.py \
 --checkpoint assets/checkpoints/go2_three_camera.pt \
 --plan configs/evaluation_routes.json --seconds 90 --name evaluation
```

此脚本只评每条路线的第一个目标；连续多目标使用巡航演示。`--no-recovery`用于关闭显式脱困的对照。所有指标都应结合成功、碰撞、跌倒和超时一起判断。

输入输出的确切形状、单位、相机参数见[INPUT_OUTPUT.md](INPUT_OUTPUT.md)。这里只提供仿真复现，没有三相机实机标定或安全验证。
