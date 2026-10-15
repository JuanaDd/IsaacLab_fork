# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched closed-loop Unitree H2 imitation using Isaac Lab's VBD lifecycle."""

from __future__ import annotations

import newton
import numpy as np
import torch
import warp as wp

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
    yaw_quat,
)

from . import mdp
from .assets import (
    ACTUATED_JOINTS,
    DEFAULT_JOINT_POS,
    MJ_BODIES,
    POLICY_JOINTS,
    POLICY_TO_MJ,
    SOFT_JOINT_LIMIT_FACTOR,
    TRACKED_BODIES,
    actuator_params,
    build_h2,
)
from .configs import H2MimicEnvCfg
from .physics import H2VBDManager


class H2MimicEnv(DirectRLEnv):
    """Use maximal-coordinate resets to preserve all six closed kinematic loops."""

    cfg: H2MimicEnvCfg

    def __init__(self, cfg: H2MimicEnvCfg, **kwargs):
        if not cfg.motion_file:
            raise ValueError("Set H2_MOTION_FILE or env.motion_file to the 50 Hz H2 motion NPZ.")
        if cfg.compute_final_obs:
            raise ValueError("The H2 reference command currently requires compute_final_obs=False.")
        super().__init__(cfg, **kwargs)
        self.model = H2VBDManager.get_model()
        self.body_count = self.model.body_count // self.num_envs
        self.coord_count = self.model.joint_coord_count // self.num_envs
        self.dof_count = self.model.joint_dof_count // self.num_envs
        labels = [label.rsplit("/", 1)[-1] for label in self.model.body_label[: self.body_count]]
        joints = [label.rsplit("/", 1)[-1] for label in self.model.joint_label]
        self.body_indices = [labels.index(name) for name in MJ_BODIES]
        joint_indices = [joints.index(name) for name in POLICY_JOINTS]
        self.coord_indices = self.model.joint_q_start.numpy()[joint_indices].tolist()
        self.dof_indices = self.model.joint_qd_start.numpy()[joint_indices].tolist()
        self.tracked = [MJ_BODIES.index(name) for name in TRACKED_BODIES]
        lower = wp.to_torch(self.model.joint_limit_lower)[self.dof_indices]
        upper = wp.to_torch(self.model.joint_limit_upper)[self.dof_indices]
        midpoint = (lower + upper) * 0.5
        half_range = (upper - lower) * (0.5 * SOFT_JOINT_LIMIT_FACTOR)
        self.soft_joint_lower = midpoint - half_range
        self.soft_joint_upper = midpoint + half_range
        self.contact_history = wp.to_torch(H2VBDManager.contact_force_history).view(
            self.num_envs, self.body_count, 3, 3
        )
        self.undesired_contact_indices = [
            i for i, name in enumerate(labels) if "ankle" not in name and "wrist_yaw" not in name
        ]
        self.ee_indices = [
            MJ_BODIES.index(name)
            for name in (
                "left_ankle_pitch_link",
                "right_ankle_pitch_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            )
        ]
        self.default_q = torch.tensor([DEFAULT_JOINT_POS.get(name, 0.0) for name in POLICY_JOINTS], device=self.device)
        self.act_indices = [POLICY_JOINTS.index(name) for name in ACTUATED_JOINTS]
        self.action_scale = torch.tensor([actuator_params(name)[3] for name in ACTUATED_JOINTS], device=self.device)
        self.actions = torch.zeros(self.num_envs, 31, device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.encoder_bias = torch.zeros(self.num_envs, 37, device=self.device)
        if cfg.randomize:
            self.encoder_bias.uniform_(-0.01, 0.01)
        self.frames = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.push_countdown = torch.empty(self.num_envs, device=self.device).uniform_(1.0, 3.0)
        self._advance_pending = False
        self._load_motion()
        self.bin_count = self.num_frames // 50 + 1
        self.bin_failures = torch.zeros(self.bin_count, device=self.device)
        self.pending_failures = torch.zeros_like(self.bin_failures)
        self.aligned_pose = torch.zeros(self.num_envs, 38, 7, device=self.device)
        self.episode_sums = {name: torch.zeros(self.num_envs, device=self.device) for name in mdp.REWARD_WEIGHTS}
        self._refresh_state()

    def _setup_scene(self) -> None:
        template = build_h2(self.cfg.mjcf_path, parse_visuals=self.cfg.parse_visuals)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.replicate(template, self.num_envs)
        if self.cfg.randomize:
            rng = np.random.default_rng(self.cfg.seed)
            for body, label in enumerate(builder.body_label):
                if label.rsplit("/", 1)[-1] == "torso_link":
                    offset = rng.uniform([-0.025, -0.05, -0.05], [0.025, 0.05, 0.05])
                    builder.body_com[body] = wp.vec3(np.asarray(builder.body_com[body]) + offset)
            friction = rng.uniform(0.3, 1.2, self.num_envs)
            for shape, label in enumerate(builder.shape_label):
                # Newton mixes the two materials; set both foot and floor to
                # the sampled coefficient, matching MuJoCo's foot priority.
                if ("foot_" in label and "collision" in label) or builder.shape_body[shape] < 0:
                    builder.shape_material_mu[shape] = float(friction[builder.shape_world[shape]])
        builder.color(balance_colors=False)
        H2VBDManager.set_builder(builder)

    def _load_motion(self) -> None:
        with np.load(self.cfg.motion_file, allow_pickle=False) as data:
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            if not np.isclose(fps * self.step_dt, 1.0):
                raise ValueError(f"Motion FPS {fps} does not match control dt {self.step_dt}")
            self.num_frames = len(data["joint_pos"])
            self.motion = {}
            for key, shape in {
                "joint_pos": (37,),
                "joint_vel": (37,),
                "body_pos_w": (38, 3),
                "body_quat_w": (38, 4),
                "body_lin_vel_w": (38, 3),
                "body_ang_vel_w": (38, 3),
            }.items():
                array = data[key]
                if array.shape != (self.num_frames, *shape) or not np.isfinite(array).all():
                    raise ValueError(f"Invalid motion array {key}: {array.shape}")
                if key in {"joint_pos", "joint_vel"}:
                    array = array[:, POLICY_TO_MJ]
                self.motion[key] = torch.as_tensor(array.copy(), dtype=torch.float32, device=self.device)
        quats = self.motion["body_quat_w"]
        if self.num_frames < 2 or not torch.allclose(quats.norm(dim=-1), torch.ones_like(quats[..., 0]), atol=1e-3):
            raise ValueError("Motion requires at least two frames and normalized wxyz quaternions.")
        self.motion["body_quat_w"] = quats[..., [1, 2, 3, 0]]
        self.reference_pose = torch.cat((self.motion["body_pos_w"], self.motion["body_quat_w"]), dim=-1)
        self.reference_velocity = torch.cat((self.motion["body_lin_vel_w"], self.motion["body_ang_vel_w"]), dim=-1)
        self.reset_pose = torch.zeros(self.num_frames, self.body_count, 7, device=self.device)
        self.reset_velocity = torch.zeros(self.num_frames, self.body_count, 6, device=self.device)
        self.reset_pose[:, self.body_indices] = self.reference_pose
        self.reset_velocity[:, self.body_indices] = self.reference_velocity
        parent = self.model.joint_parent.numpy()
        child = self.model.joint_child.numpy()
        types = self.model.joint_type.numpy()
        xp = torch.as_tensor(self.model.joint_X_p.numpy(), device=self.device)
        xc = torch.as_tensor(self.model.joint_X_c.numpy(), device=self.device)
        self.rod_joints = []
        for rod in range(self.body_count):
            ends = [j for j, c in enumerate(child) if c == rod and types[j] == int(newton.JointType.BALL)]
            if not ends:
                continue
            if len(ends) != 2:
                raise ValueError(f"Rod {rod} must have exactly two ball joints")
            self.rod_joints.extend(ends)
            points, speeds = [], []
            for j in ends:
                body_pose = self.reset_pose[:, parent[j]]
                lever = quat_apply(body_pose[:, 3:], xp[j, :3].expand(self.num_frames, -1))
                points.append(body_pose[:, :3] + lever)
                vel = self.reset_velocity[:, parent[j]]
                speeds.append(vel[:, :3] + torch.cross(vel[:, 3:], lever, dim=-1))
            vector = points[1] - points[0]
            direction = vector / vector.norm(dim=-1, keepdim=True)
            z = torch.zeros_like(direction)
            z[:, 2] = 1.0
            quat = torch.cat((torch.cross(z, direction, dim=-1), 1 + direction[:, 2:3]), dim=-1)
            quat /= quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            self.reset_pose[:, rod, 3:] = quat
            midpoint = (xc[ends[0], :3] + xc[ends[1], :3]) * 0.5
            self.reset_pose[:, rod, :3] = (points[0] + points[1]) * 0.5 - quat_apply(
                quat, midpoint.expand(self.num_frames, -1)
            )
            self.reset_velocity[:, rod, :3] = (speeds[0] + speeds[1]) * 0.5
            self.reset_velocity[:, rod, 3:] = torch.cross(vector, speeds[1] - speeds[0], dim=-1) / vector.square().sum(
                -1, keepdim=True
            )
        if len(self.rod_joints) != 12:
            raise ValueError("H2 must contain all six closed-loop rods")

    def _refresh_state(self) -> None:
        state = H2VBDManager.get_state_0()
        self.all_pose = wp.to_torch(state.body_q).view(self.num_envs, self.body_count, 7)
        self.all_velocity_com = wp.to_torch(state.body_qd).view(self.num_envs, self.body_count, 6)
        self.q = wp.to_torch(state.joint_q).view(self.num_envs, self.coord_count)[:, self.coord_indices]
        self.qd = wp.to_torch(state.joint_qd).view(self.num_envs, self.dof_count)[:, self.dof_indices]
        self.pose = self.all_pose[:, self.body_indices]
        self.velocity = self.all_velocity_com[:, self.body_indices].clone()
        com = wp.to_torch(self.model.body_com).view(self.num_envs, self.body_count, 3)[:, self.body_indices]
        self.velocity[..., :3] -= torch.cross(self.velocity[..., 3:], quat_apply(self.pose[..., 3:], com), dim=-1)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.previous_actions.copy_(self.actions)
        self.actions.copy_(actions)
        self._advance_pending = True

    def _apply_action(self) -> None:
        wp.to_torch(H2VBDManager.desired).view(self.num_envs, 31).copy_(
            self.default_q[self.act_indices] + self.encoder_bias[:, self.act_indices] + self.action_scale * self.actions
        )

    def _push(self, ids: torch.Tensor) -> None:
        delta = (2 * torch.rand(len(ids), 6, device=self.device) - 1) * torch.tensor(
            [0.5, 0.5, 0.2, 0.52, 0.52, 0.78], device=self.device
        )
        for state in (H2VBDManager.get_state_0(), H2VBDManager.get_state_1()):
            pose = wp.to_torch(state.body_q).view(self.num_envs, self.body_count, 7)
            vel = wp.to_torch(state.body_qd).view(self.num_envs, self.body_count, 6)
            lever = pose[ids, :, :3] - pose[ids, 0:1, :3]
            com = wp.to_torch(self.model.body_com).view(self.num_envs, self.body_count, 3)[ids]
            lever += quat_apply(pose[ids, :, 3:], com)
            vel[ids, :, :3] += delta[:, None, :3] + torch.cross(delta[:, None, 3:].expand_as(lever), lever, dim=-1)
            vel[ids, :, 3:] += delta[:, None, 3:]

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._refresh_state()
        ref = self.reference_pose[self.frames]
        gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        gravity_error = (
            quat_apply_inverse(ref[:, 19, 3:], gravity)[:, 2] - quat_apply_inverse(self.pose[:, 19, 3:], gravity)[:, 2]
        ).abs()
        terminated = (ref[:, 19, 2] - self.pose[:, 19, 2]).abs() > 0.25
        terminated |= gravity_error > 0.8
        terminated |= (self.aligned_pose[:, self.ee_indices, 2] - self.pose[:, self.ee_indices, 2]).abs().amax(
            -1
        ) > 0.35
        terminated |= ~torch.isfinite(self.q).all(-1) | ~torch.isfinite(self.pose).flatten(1).all(-1)
        return terminated, self.episode_length_buf >= self.max_episode_length

    def _get_rewards(self) -> torch.Tensor:
        terms = mdp.tracking_terms(
            self.q,
            self.pose,
            self.velocity,
            self.motion["joint_pos"][self.frames],
            self.reference_pose[self.frames],
            self.reference_velocity[self.frames],
            self.aligned_pose,
            self.tracked,
            self.actions,
            self.previous_actions,
        )
        terms["joint_limit"] = (
            (self.soft_joint_lower - self.q).clamp_min(0) + (self.q - self.soft_joint_upper).clamp_min(0)
        ).sum(-1)
        peak_force = self.contact_history[:, self.undesired_contact_indices].norm(dim=-1).amax(-1)
        terms["undesired_contacts"] = (peak_force > 1.0).sum(-1)
        total = torch.zeros(self.num_envs, device=self.device)
        for name, value in terms.items():
            weighted = value * mdp.REWARD_WEIGHTS[name] * self.step_dt
            self.episode_sums[name] += weighted
            total += weighted
        # mjlab applies interval events after reward/termination evaluation,
        # before auto-reset and before constructing the next observation.
        if self.cfg.randomize:
            self.push_countdown -= self.step_dt
            ids = (self.push_countdown <= 0).nonzero(as_tuple=False).flatten()
            if ids.numel():
                self._push(ids)
                self.push_countdown[ids] = torch.empty(len(ids), device=self.device).uniform_(1.0, 3.0)
        return total

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        ids = env_ids.long()
        self.extras["log"] = {
            f"Episode_Reward/{name}": sums[ids].mean() / self.cfg.episode_length_s
            for name, sums in self.episode_sums.items()
        }
        for sums in self.episode_sums.values():
            sums[ids] = 0
        self._resample_motion(ids)
        self.actions[ids] = 0
        self.previous_actions[ids] = 0
        if self.cfg.randomize:
            self.push_countdown[ids] = torch.empty(len(ids), device=self.device).uniform_(1.0, 3.0)
        self._advance_pending = True
        super()._reset_idx(env_ids)

    def _resample_motion(self, ids: torch.Tensor) -> None:
        """Resample reference state without resetting the episode or action history."""
        if self.cfg.sampling_mode == "adaptive":
            failed = ids[self.reset_terminated[ids]]
            fail_bins = (self.frames[failed] * self.bin_count // self.num_frames).clamp_max(self.bin_count - 1)
            if failed.numel():
                self.pending_failures.copy_(torch.bincount(fail_bins, minlength=self.bin_count))
            probabilities = self.bin_failures + 0.1 / self.bin_count
            bins = torch.multinomial(probabilities, len(ids), replacement=True)
            self.frames[ids] = (
                (bins + torch.rand(len(ids), device=self.device)) / self.bin_count * (self.num_frames - 1)
            ).long()
        elif self.cfg.sampling_mode == "uniform":
            self.frames[ids] = torch.randint(self.num_frames, (len(ids),), device=self.device)
        elif self.cfg.sampling_mode == "start":
            self.frames[ids] = 0
        else:
            raise ValueError(f"Unknown sampling mode {self.cfg.sampling_mode}")
        self._write_reference_state(ids)

    def _reset_envs_from_buffer(self) -> torch.Tensor | None:
        if self.cfg.auto_reset:
            return super()._reset_envs_from_buffer()
        return torch.empty(0, dtype=torch.long, device=self.device)

    def _write_reference_state(self, ids: torch.Tensor) -> None:
        self.contact_history[ids] = 0
        pose = self.reset_pose[self.frames[ids]].clone()
        velocity = self.reset_velocity[self.frames[ids]].clone()
        if self.cfg.randomize:
            root_velocity = velocity[:, 0].clone()
            pose_noise = (2 * torch.rand(len(ids), 6, device=self.device) - 1) * torch.tensor(
                [0.05, 0.05, 0.01, 0.1, 0.1, 0.2], device=self.device
            )
            dq = quat_from_euler_xyz(*pose_noise[:, 3:].unbind(-1))[:, None].expand(-1, self.body_count, -1)
            origin = pose[:, 0:1, :3].clone()
            pose[..., :3] = origin + quat_apply(dq, pose[..., :3] - origin) + pose_noise[:, None, :3]
            pose[..., 3:] = quat_mul(dq, pose[..., 3:])
            velocity[..., :3] = quat_apply(dq, velocity[..., :3])
            velocity[..., 3:] = quat_apply(dq, velocity[..., 3:])
            delta = (2 * torch.rand(len(ids), 6, device=self.device) - 1) * torch.tensor(
                [0.5, 0.5, 0.2, 0.52, 0.52, 0.78], device=self.device
            )
            # mjlab perturbs root orientation while preserving the original
            # world-frame root twist before adding velocity noise.
            delta += root_velocity - velocity[:, 0]
            lever = pose[..., :3] - pose[:, 0:1, :3]
            velocity[..., :3] += delta[:, None, :3] + torch.cross(delta[:, None, 3:].expand_as(lever), lever, dim=-1)
            velocity[..., 3:] += delta[:, None, 3:]
        com = wp.to_torch(self.model.body_com).view(self.num_envs, self.body_count, 3)[ids]
        velocity[..., :3] += torch.cross(velocity[..., 3:], quat_apply(pose[..., 3:], com), dim=-1)
        for state in (H2VBDManager.get_state_0(), H2VBDManager.get_state_1()):
            wp.to_torch(state.body_q).view(self.num_envs, self.body_count, 7)[ids] = pose
            wp.to_torch(state.body_qd).view(self.num_envs, self.body_count, 6)[ids] = velocity
            newton.eval_ik(self.model, state, state.joint_q, state.joint_qd)
        H2VBDManager.invalidate_body_state(env_ids=wp.from_torch(ids.to(torch.int32), dtype=wp.int32))

    def _get_observations(self) -> dict[str, torch.Tensor]:
        if self._advance_pending:
            self.frames += 1
            wrap = (self.frames >= self.num_frames).nonzero(as_tuple=False).flatten()
            if wrap.numel():
                # mjlab resamples the reference and teleports on clip wrap.
                self._resample_motion(wrap)
            self.bin_failures.mul_(0.999).add_(self.pending_failures, alpha=0.001)
            self.pending_failures.zero_()
            self._advance_pending = False
        self._refresh_state()
        ref = self.reference_pose[self.frames]
        delta_q = yaw_quat(quat_mul(self.pose[:, 19, 3:], quat_conjugate(ref[:, 19, 3:])))
        delta_q = delta_q[:, None].expand(-1, 38, -1)
        delta_pos = self.pose[:, 19:20, :3].clone()
        delta_pos[..., 2] = ref[:, 19:20, 2]
        self.aligned_pose[..., :3] = delta_pos + quat_apply(delta_q, ref[..., :3] - ref[:, 19:20, :3])
        self.aligned_pose[..., 3:] = quat_mul(delta_q, ref[..., 3:])
        return mdp.observations(
            self.q,
            self.qd,
            self.pose,
            self.velocity,
            self.motion["joint_pos"][self.frames],
            self.motion["joint_vel"][self.frames],
            ref,
            self.tracked,
            self.default_q,
            self.actions,
            self.encoder_bias,
            self.cfg.observation_noise,
        )
