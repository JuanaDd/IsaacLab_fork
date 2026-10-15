# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H2 mimic configuration aligned with the original IsaacLab-mimic policy."""

from __future__ import annotations

import os

from isaaclab_newton.physics import NewtonCfg, VBDSolverCfg

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class H2VBDSolverCfg(VBDSolverCfg):
    """Fixed sparse VBD solver used by the H2 checkpoints."""

    class_type: str = "isaaclab_tasks.contrib.h2_mimic.physics:NewtonH2VBDManager"
    iterations: int = 8
    relaxation: float = 0.65


@configclass
class H2MimicEnvCfg(DirectRLEnvCfg):
    """50 Hz policy, 5 ms VBD substeps, and original IsaacLab-mimic joint ordering."""

    seed: int = 42
    decimation: int = 1
    episode_length_s: float = 10.0
    action_space: int = 31
    observation_space: int = 194
    state_space: int = 320
    sim: SimulationCfg = SimulationCfg(
        dt=0.02,
        render_interval=1,
        default_visualizer_cfg=VisualizerCfg(eye=(4.0, -4.0, 2.5), lookat=(1.0, 0.0, 1.0), focal_length=20.0),
        physics=NewtonCfg(solver_cfg=H2VBDSolverCfg(), num_substeps=4, collision_decimation=1, use_cuda_graph=True),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=0.0)
    mjcf_path: str = os.environ.get("H2_MJCF_FILE", ".local/h2_assets/H2loop_complete.xml")
    motion_file: str = os.environ.get("H2_MOTION_FILE", "")
    parse_visuals: bool = False
    """Import visual meshes for interactive playback or video recording."""
    randomize: bool = True
    sampling_mode: str = "adaptive"
    observation_noise: bool = True
    auto_reset: bool = True

    def play_mode(self) -> None:
        """Disable task-local perturbations for the standard Isaac Lab play command."""
        super().play_mode()
        self.randomize = False
        self.sampling_mode = "start"
        self.observation_noise = False
        self.episode_length_s = 1.0e9
        self.scene.num_envs = 1
        self.auto_reset = False
        self.parse_visuals = True


@configclass
class H2MimicPlayEnvCfg(H2MimicEnvCfg):
    """Deterministic single-environment playback with visual meshes."""

    episode_length_s: float = 1.0e9
    randomize: bool = False
    sampling_mode: str = "start"
    observation_noise: bool = False
    auto_reset: bool = False
    parse_visuals: bool = True
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=0.0)


@configclass
class H2MimicPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO architecture compatible with the existing H2 VBD checkpoints."""

    seed = 42
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    init_at_random_ep_len = False
    experiment_name = "h2_closeloop_mimic_vbd"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = None
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
