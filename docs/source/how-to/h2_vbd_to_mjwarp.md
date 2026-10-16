---
orphan: true
---

# H2 VBD → native MJWarp sim2sim

使用 [H2 VBD mimic](h2_mimic_vbd.md) 训练的 checkpoint，在原生 mjlab H2 模型中闭环播放。
策略每一步读取 MJWarp 当前状态，计算新的动作；使用原生环境、驱动和六组 tendon equality。

## 任务和接口

- 源任务：`IsaacContrib-H2-Closeloop-Mimic-VBD`，Isaac Lab Direct 环境。
- 目标任务：`Mjlab-Tracking-Flat-Unitree-H2`，原生 `ManagerBasedRlEnv`。
- 目标物理模型为 38 个机器人刚体、37 个关节、31 个 actuator，保留六组 XML tendon equality。
  `solref=(0.005, 1)`，`solimp=(0.9, 0.95, 0.001, 0.5, 2)`；脚本会核对 CPU/GPU 参数。
- 两端控制周期均为 20 ms，每次控制包含四个 5 ms 物理子步。
  VBD 使用 8 iterations；目标保留原生 implicitfast、Newton solver 100 iterations、
  50 line-search iterations 和 tolerance `1e-8`。
- 目标保留原生 PD、阻尼、effort 限制、关节限位和摩擦设置。
  两端仍有刚体表示和驱动差异，运行成功不代表动力学完全一致。
- 仅按关节名称重排 194 维 actor 观测及 31 维动作。共享的
  `h2_mimic/joints.py` 同时供 VBD 资产和本脚本使用，无需 `PYTHONPATH`。
- 复用 mjlab/RSL-RL 的 actor 和 normalizer 加载；不加载维度不同的 critic，不更新权重或归一化统计。

## 固定依赖

源端依赖为 Newton **1.6.0**，提交 `c2ca70bec5998062b0b2d35869865cbe3a49feee`，
加 `scripts/h2/newton_vbd.patch`；安装方法见源任务文档。

目标端使用独立环境，保持原生参考实现的版本：

| 依赖 | 固定版本 |
|---|---|
| mjlab | 1.6.0，`https://github.com/JuanaDd/mjlab.git` |
| mjlab 提交 | `ba3f5814a703e6823513aac4a44d0f80c29a2284` |
| Python | 3.13 |
| Torch | 2.9.0 + CUDA 12.8 |
| Warp | `1.16.0.dev20260716` |
| MuJoCo / MuJoCo Warp | 3.11.0 / 3.11.0 |
| RSL-RL | 5.5.0 |

在 Isaac Lab 仓库根目录执行：

```bash
bash scripts/h2/setup_mjlab.sh
```

脚本将固定提交安装到 `.local/mjlab_h2`，按该提交的 lockfile 创建独立虚拟环境，再固定参考 Warp 版本。
已有目录会拒绝覆盖；`MJLAB_REPOSITORY` 可指定包含该提交的 Git 镜像或本地仓库。
运行时使用 `--no-project` 指定解释器，避免重新同步或替换 Warp。
目标环境不依赖 Newton fork、mjlab 以外的参考仓库或旧 benchmark 文件。

## 资产和权重

复用源任务的本地文件：

```text
.local/h2_assets/H2loop_complete.xml
.local/h2_assets/meshes/
.local/h2_data/h2_360dankou.npz
.pretrained_checkpoints/h2/model_45500.pt
```

XML 的 mesh 相对引用必须完整保留。资产、checkpoint、运行输出以及 MP4 均不提交；
已有 `.gitignore` 规则覆盖 XML、MP4、`.local/`、`.pretrained_checkpoints/` 和 `artifacts/`。

## 播放与录像

在有桌面显示的会话中打开原生 MuJoCo viewer：

```bash
export CUDA_VISIBLE_DEVICES=0

uv run --no-project --python .local/mjlab_h2/.venv/bin/python python scripts/h2/sim2sim.py \
  --checkpoint .pretrained_checkpoints/h2/model_45500.pt \
  --motion .local/h2_data/h2_360dankou.npz \
  --viewer
```

无窗口录制视频：

```bash
MUJOCO_GL=egl uv run --no-project --python .local/mjlab_h2/.venv/bin/python \
  python scripts/h2/sim2sim.py \
  --checkpoint .pretrained_checkpoints/h2/model_45500.pt \
  --motion .local/h2_data/h2_360dankou.npz \
  --video_dir artifacts/h2_vbd_mjwarp/model_45500
```

输出为 `artifacts/h2_vbd_mjwarp/model_45500/model_45500-step-0.mp4`，1280×720、50 fps。
再次运行需要选择新的输出目录。`--viewer` 与 `--video_dir` 可同时使用；均省略时只运行闭环评估。
`--mjcf` 可指定其他资产位置，`--steps` 可缩短播放，默认最多 476 步 / 9.52 s，避免动作片段末尾重采样。

**脚本关闭自动重置，并在首次原生 done 时停止播放和录制。**
原生末端高度阈值为 0.25 m，源任务为 0.35 m；目标保留其原生阈值。
因此录像可能短于 9.52 s；终止表示触发任务判据，不直接等同于倒地。
结束时打印实际步数、模拟时长、首次终止步和包版本，便于核对结果。

本次验证中，`model_45500.pt` 在原生判据下于第 269 步 / **5.38 s** 终止。
同一文件在 VBD 源环境中完成 476 步 / 9.52 s。
`model_33000.pt` 的目标端终止时间仍为 5.42 s，与此前原生 mjlab 实现一致。

## 实现位置

- `scripts/h2/sim2sim.py`：原生闭环播放、actor 加载、viewer 和视频入口。
- `scripts/h2/setup_mjlab.sh`：独立目标环境安装。
- `source/isaaclab_tasks/isaaclab_tasks/contrib/h2_mimic/joints.py`：共享关节顺序及观测/动作映射。
- `source/isaaclab_tasks/test/contrib/test_h2_mdp.py`：用源 MDP 独立验证重排后的观测和动作。
