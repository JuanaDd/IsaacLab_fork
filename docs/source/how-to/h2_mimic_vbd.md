---
orphan: true
---

# H2 motion imitation with Newton VBD

本任务使用 Isaac Lab 3.0 的 **DirectRLEnv** 和 RSL-RL PPO，在 Newton VBD 中训练
Unitree H2 跟踪参考动作。训练、恢复和播放均使用 Isaac Lab CLI。

## 任务与控制配置

| 项目 | 配置 |
|---|---|
| 训练任务 | `IsaacContrib-H2-Closeloop-Mimic-VBD` |
| 确定性评估任务 | `IsaacContrib-H2-Closeloop-Mimic-VBD-Eval` |
| 物理后端 | Newton VBD，`block_sparse_joints`，CUDA Graph |
| 控制周期 | 20 ms，每次控制包含 4 个 5 ms 物理子步 |
| 求解 | 每个子步固定 8 iterations，relaxation 0.65 |
| 训练并行数 | 默认 8192，支持 `--num_envs` 覆盖 |
| 机器人 | 38 个原始刚体，加 6 根具有质量的推杆；每根推杆两端各一个球关节 |
| 动作 | 31 维关节位置偏移，PD 在每个物理子步更新 |
| 观测 | actor 194 维，critic 320 维；关节按原 H2 checkpoint 顺序排列 |
| 网络 | actor/critic 均为 512–256–128，ELU，观测归一化 |

奖励包含 torso 位姿、身体位姿和速度跟踪、动作变化、关节限位及非期望接触惩罚。
头部另外包含关节位置、刚体位置和刚体方向跟踪奖励，权重各为 0.5。
头部奖励没有改变 actor/critic 维度，已有 checkpoint 可以继续训练或播放。

训练保留参考动作自适应采样、初始状态扰动、外部推力、摩擦和质心随机化。
play 从动作起点开始，关闭随机化和观测噪声，启用完整视觉网格，默认只播放一个环境。
play 关闭失败自动重置；动作片段结束时会重新采样参考起点并重置机器人。

## 依赖与安装

在仓库根目录执行本文命令。Python 和包版本由 `uv.lock` 管理；验证环境为
Python 3.12、PyTorch 2.11.0 + CUDA 12.8、Warp 1.17.0、RSL-RL 5.5.1。
训练使用 NVIDIA CUDA GPU，已在 RTX PRO 5000 上验证。

**Newton 依赖是上游 1.6.0 加本仓库随附的 H2 VBD 补丁**：

- 上游仓库：`https://github.com/newton-physics/newton.git`。
- 固定提交：`c2ca70bec5998062b0b2d35869865cbe3a49feee`（`v1.6.0`）。
- 补丁：`scripts/h2/newton_vbd.patch`，仅修改 VBD 运行时代码，提供闭环稀疏关节求解、
  joint armature、接触历史和训练所用的 CUDA 稀疏求解实现。
- 安装位置：`.local/newton_h2_lab30`，通过 `pyproject.toml` 中的 uv source 引用。
  仅安装同版本的 PyPI Newton 包不足以提供本任务使用的稀疏闭环求解接口。

```bash
bash scripts/h2/setup_newton.sh
uv sync --extra video
```

安装脚本固定 checkout 上述提交并应用补丁；若目标目录已经存在则拒绝覆盖。
可以用 `NEWTON_REPOSITORY` 指向包含该上游提交的 Git 镜像或已有本地仓库。
不需要 mjlab、其他 Newton fork 或参考实验目录。

## 本地资产和 checkpoint

资产、动作和权重需自行提供，按以下相对目录放置：

```text
.local/
  h2_assets/
    H2loop_complete.xml
    meshes/                 # 保留 XML 引用的目录结构和文件名
  h2_data/
    h2_360dankou.npz
.pretrained_checkpoints/
  h2/
    model_45500.pt
```

MJCF 必须包含左右膝、左右踝和腰部的六组推杆连接 site；构建器将其重建为刚性推杆。
应复制完整 H2 资产目录，保留 XML 的 mesh 相对引用。XML 资产和 MP4 文件已加入
`.gitignore`；本地依赖、日志、动作、checkpoint 和录像也不提交。

动作 NPZ 使用 50 Hz，并包含以下数组；浮点数据必须有限，四元数必须归一化。

| 键 | 形状 / 约定 |
|---|---|
| `fps` | 标量或单元素数组，值为 50 |
| `joint_pos` / `joint_vel` | `[T, 37]`，MJCF 关节顺序，单位 rad / rad/s |
| `body_pos_w` | `[T, 38, 3]`，MJCF 刚体顺序，单位 m |
| `body_quat_w` | `[T, 38, 4]`，输入为 wxyz；环境内部转为 xyzw |
| `body_lin_vel_w` / `body_ang_vel_w` | `[T, 38, 3]`，世界坐标系，单位 m/s / rad/s |

```bash
export H2_MJCF_FILE=.local/h2_assets/H2loop_complete.xml
export H2_MOTION_FILE=.local/h2_data/h2_360dankou.npz
export CUDA_VISIBLE_DEVICES=0
```

也可通过 Hydra 参数 `env.mjcf_path=...` 和 `env.motion_file=...` 指定其他位置。
以下命令沿用上述环境变量。

## Train

先执行少量环境和迭代的训练，检查资产、CUDA 和 PPO 链路：

```bash
uv run isaaclab train --task IsaacContrib-H2-Closeloop-Mimic-VBD \
  --device cuda:0 --num_envs 32 --max_iterations 2 --run_name h2_smoke
```

从头训练，使用 8192 个环境和默认 5 ms / 8 iterations：

```bash
uv run isaaclab train --task IsaacContrib-H2-Closeloop-Mimic-VBD \
  --device cuda:0 --num_envs 8192 --max_iterations 30000 --logger tensorboard
```

从已有 checkpoint 继续训练：

```bash
uv run isaaclab train --task IsaacContrib-H2-Closeloop-Mimic-VBD \
  --device cuda:0 --num_envs 8192 --max_iterations 30000 \
  --checkpoint .pretrained_checkpoints/h2/model_45500.pt --logger tensorboard
```

恢复会加载 actor、critic、观测归一化、优化器和迭代号；`--max_iterations` 是本次追加的更新次数。
日志和每 500 次更新的 checkpoint 保存在 `logs/rsl_rl/h2_closeloop_mimic_vbd/` 下。

## Play 与录制视频

使用本地最新已训练 checkpoint 打开 Newton OpenGL 可视化：

```bash
uv run isaaclab play --task IsaacContrib-H2-Closeloop-Mimic-VBD \
  --device cuda:0 --num_envs 1 --checkpoint .pretrained_checkpoints/h2/model_45500.pt \
  --viz newton_gl --real-time
```

录制一个完整动作片段并自动退出：

```bash
uv run --extra video isaaclab play --task IsaacContrib-H2-Closeloop-Mimic-VBD \
  --device cuda:0 --num_envs 1 --checkpoint .pretrained_checkpoints/h2/model_45500.pt \
  --viz newton_gl --video --video_length 476
```

配套 `h2_360dankou.npz` 为 478 帧；476 个控制步覆盖起点后的 9.52 s，避开片段末尾重采样。
录像保存在 checkpoint 所在目录的 `videos/play/` 下。
没有显示服务器时 Newton GL 自动通过 EGL 离屏渲染；桌面会话中可直接交互查看。
以后训练出的模型可替换 `--checkpoint` 路径；`--checkpoint latest` 会选择最新本地训练目录中的权重，
因此存在 smoke run 时建议明确指定要评估的 checkpoint。

本分支完成了 32 环境各 2 次更新的从头/恢复训练检查，以及 8192 环境 1 次更新的恢复检查。
`model_45500.pt` 在默认 5 ms / 8 iterations 下完成 476 步 / 9.52 s，未触发任务终止阈值；
最大末端高度误差为 0.2723 m（阈值 0.35 m）。
Newton GL 录制的视频为 1920×1080、50 fps，全片解码通过。

## 实现位置

- `source/isaaclab_tasks/isaaclab_tasks/contrib/h2_mimic/`：任务注册、机器人构建、
  Direct 环境、观察与奖励、VBD manager 和 PPO 配置。
- `scripts/h2/setup_newton.sh`：固定版本 Newton 安装。
- `scripts/h2/newton_vbd.patch`：所需 Newton VBD 扩展。
- `source/isaaclab_tasks/test/contrib/test_h2_mdp.py`：独立验证头部奖励作用于头部。

训练和播放不使用额外的 benchmark、sim2sim 或分析 runner。

将 VBD checkpoint 部署到原生 MJWarp，见 [H2 VBD → native MJWarp sim2sim](h2_vbd_to_mjwarp.md)。
