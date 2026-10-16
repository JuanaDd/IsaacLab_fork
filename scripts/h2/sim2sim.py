# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a Newton VBD H2 checkpoint in the original mjlab tendon model.

Run with the isolated mjlab Python environment prepared by setup_mjlab.sh.
Native physics, gains, limits and termination thresholds are preserved.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer
from tensordict import TensorDict


class _VbdPolicyEnv(RslRlVecEnvWrapper):
    """Expose native mjlab feedback in the VBD checkpoint's column convention."""

    def __init__(self, env: ManagerBasedRlEnv):
        # Load only the shared joint contract, without importing the Newton task.
        source = Path(__file__).resolve().parents[2] / "source/isaaclab_tasks/isaaclab_tasks/contrib/h2_mimic/joints.py"
        spec = importlib.util.spec_from_file_location("h2_policy_joints", source)
        joints = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(joints)
        self.actor_columns, self.action_columns = joints.policy_permutations(
            env.scene["robot"].joint_names, env.action_manager.get_term("joint_pos").target_names
        )
        self.steps = 0
        self.first_termination_step: int | None = None
        super().__init__(env, clip_actions=None)

    def get_observations(self) -> TensorDict:
        return self._map_observations(super().get_observations())

    def reset(self) -> tuple[TensorDict, dict]:
        obs, extras = super().reset()
        return self._map_observations(obs), extras

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        obs, reward, done, extras = super().step(actions[:, self.action_columns])
        self.steps += 1
        if bool(done.any()) and self.first_termination_step is None:
            self.first_termination_step = self.steps
        if not torch.isfinite(obs["actor"]).all():
            raise FloatingPointError(f"Non-finite policy observation at step {self.steps}")
        return self._map_observations(obs), reward, done, extras

    def _map_observations(self, obs: TensorDict) -> TensorDict:
        if obs["actor"].shape[-1] != 194:
            raise ValueError("Expected the native H2 task's 194-dimensional actor observation.")
        obs["actor"] = obs["actor"][:, self.actor_columns]
        return obs


class _TransferViewer(NativeMujocoViewer):
    """Stop before mjlab forbids another step after a terminated episode."""

    env: _VbdPolicyEnv

    def is_running(self) -> bool:
        return super().is_running() and self.env.first_termination_step is None

    def _execute_step(self) -> bool:
        return super()._execute_step() and self.env.first_termination_step is None


def main() -> None:
    """Load the actor, run closed-loop native MJWarp playback, and optionally record it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, default=Path(".local/h2_assets/H2loop_complete.xml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=476)
    parser.add_argument("--viewer", action="store_true", help="Open the native MuJoCo viewer.")
    parser.add_argument("--video_dir", type=Path, help="Record a video using mjlab's offscreen renderer.")
    args = parser.parse_args()
    for path in (args.checkpoint, args.motion, args.mjcf):
        if not path.is_file():
            parser.error(f"Input file not found: {path}")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.video_dir and (args.video_dir / f"{args.checkpoint.stem}-step-0.mp4").exists():
        parser.error("The output video already exists; select a new --video_dir")
    os.environ["MJLAB_H2_XML"] = str(args.mjcf.resolve())

    from mjlab.tasks.tracking.config.h2.env_cfgs import unitree_h2_flat_tracking_env_cfg
    from mjlab.tasks.tracking.config.h2.rl_cfg import unitree_h2_tracking_ppo_runner_cfg

    cfg = unitree_h2_flat_tracking_env_cfg(play=True)
    cfg.auto_reset = False
    cfg.seed = args.seed
    cfg.commands["motion"].motion_file = str(args.motion.resolve())
    cfg.viewer.width, cfg.viewer.height = 1280, 720
    raw_env = ManagerBasedRlEnv(cfg, device=args.device, render_mode="rgb_array" if args.video_dir else None)
    env = None
    viewer = None
    try:
        model = raw_env.sim.mj_model
        if len(raw_env.scene["robot"].body_names) != 38 or not np.all(model.eq_type == mujoco.mjtEq.mjEQ_TENDON):
            raise ValueError("Expected the native 38-body H2 model with tendon equalities.")
        np.testing.assert_allclose(model.eq_solimp, np.tile([0.9, 0.95, 0.001, 0.5, 2], (6, 1)))
        np.testing.assert_allclose(model.eq_solref, np.tile([0.005, 1], (6, 1)))
        np.testing.assert_allclose(raw_env.sim.wp_model.eq_solimp.numpy()[0], model.eq_solimp, atol=1e-6)
        np.testing.assert_allclose(raw_env.sim.wp_model.eq_solref.numpy()[0], model.eq_solref, atol=1e-6)
        if args.steps > raw_env.command_manager.get_term("motion").motion.time_step_total - 2:
            parser.error("--steps would reach the reference motion's end and reset the robot")
        if args.video_dir:
            raw_env = VideoRecorder(
                raw_env,
                video_folder=args.video_dir,
                step_trigger=lambda step: step == 0,
                video_length=args.steps,
                name_prefix=args.checkpoint.stem,
            )
        env = _VbdPolicyEnv(raw_env)
        runner = MjlabOnPolicyRunner(env, asdict(unitree_h2_tracking_ppo_runner_cfg()), device=args.device)
        runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        if args.viewer:
            viewer = _TransferViewer(env, policy)
            viewer.run(num_steps=args.steps)
        else:
            obs = env.get_observations()
            with torch.inference_mode():
                for _ in range(args.steps):
                    obs, _, done, _ = env.step(policy(obs))
                    if bool(done.any()):
                        break
        print(
            json.dumps(
                {
                    "checkpoint": args.checkpoint.name,
                    "steps": env.steps,
                    "simulated_seconds": env.steps * cfg.decimation * cfg.sim.mujoco.timestep,
                    "first_native_termination_step": env.first_termination_step,
                    "auto_reset": cfg.auto_reset,
                    "packages": {
                        name: importlib.metadata.version(name)
                        for name in ("mjlab", "torch", "warp-lang", "mujoco", "mujoco-warp", "rsl-rl-lib")
                    },
                },
                indent=2,
            )
        )
    finally:
        if viewer is not None:
            viewer.close()
        (env if env is not None else raw_env).close()


if __name__ == "__main__":
    main()
