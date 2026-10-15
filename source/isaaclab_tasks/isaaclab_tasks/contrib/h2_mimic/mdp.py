# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H2 observations and tracking rewards for motion imitation.

All quaternions in this module use Isaac Lab 3 / Newton xyzw ordering.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, quat_conjugate, quat_mul


def rotation_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the shortest orientation error [rad] between xyzw quaternions."""
    relative = quat_mul(a, quat_conjugate(b))
    return 2.0 * torch.atan2(relative[..., :3].norm(dim=-1), relative[..., 3].abs())


def observations(
    q: torch.Tensor,
    qd: torch.Tensor,
    pose: torch.Tensor,
    velocity: torch.Tensor,
    ref_q: torch.Tensor,
    ref_qd: torch.Tensor,
    ref_pose: torch.Tensor,
    tracked: list[int],
    default_q: torch.Tensor,
    action: torch.Tensor,
    encoder_bias: torch.Tensor,
    noise: bool,
) -> dict[str, torch.Tensor]:
    """Build the original 194-dimensional actor and 320-dimensional critic.

    Args:
        q: Joint positions [rad].
        qd: Joint velocities [rad/s].
        pose: Body positions [m] and xyzw orientations.
        velocity: World-frame body linear [m/s] and angular [rad/s] velocities.
        ref_q: Reference joint positions [rad].
        ref_qd: Reference joint velocities [rad/s].
        ref_pose: Reference body positions [m] and xyzw orientations.
        tracked: Body indices included in the critic observations.
        default_q: Nominal joint positions [rad].
        action: Previous normalized policy action.
        encoder_bias: Per-environment joint encoder offsets [rad].
        noise: Whether to corrupt the actor observations during training.

    Returns:
        Policy and critic observations with batch dimension first.
    """
    anchor = 19  # torso_link in the explicit 38-body motion contract
    anchor_q = pose[:, anchor, 3:]
    relative_q = quat_mul(quat_conjugate(anchor_q), ref_pose[:, anchor, 3:])
    anchor_pos = quat_apply_inverse(anchor_q, ref_pose[:, anchor, :3] - pose[:, anchor, :3])
    anchor_ori = matrix_from_quat(relative_q)[..., :2].flatten(1)
    lin_vel = quat_apply_inverse(pose[:, 0, 3:], velocity[:, 0, :3])
    ang_vel = quat_apply_inverse(pose[:, 0, 3:], velocity[:, 0, 3:])
    # The original startup event shifts both default-position observations and
    # action offsets by the same per-environment encoder bias.
    joint_pos = q - default_q - encoder_bias
    command = torch.cat((ref_q, ref_qd), dim=-1)
    pieces = (anchor_pos, anchor_ori, lin_vel, ang_vel, joint_pos, qd)
    noisy = list(pieces)
    if noise:
        noisy = [
            term + (2 * torch.rand_like(term) - 1) * scale
            for term, scale in zip(pieces, (0.25, 0.05, 0.5, 0.2, 0.01, 0.5), strict=True)
        ]
    actor = torch.cat((command, *noisy, action), dim=-1)
    aq = anchor_q[:, None].expand(-1, len(tracked), -1)
    body_pos = quat_apply_inverse(aq, pose[:, tracked, :3] - pose[:, anchor : anchor + 1, :3]).flatten(1)
    body_ori = matrix_from_quat(quat_mul(quat_conjugate(aq), pose[:, tracked, 3:]))[..., :2].flatten(1)
    critic = torch.cat(
        (command, anchor_pos, anchor_ori, body_pos, body_ori, lin_vel, ang_vel, joint_pos, qd, action), dim=-1
    )
    return {"policy": actor, "critic": critic}


def tracking_terms(
    q: torch.Tensor,
    pose: torch.Tensor,
    velocity: torch.Tensor,
    ref_q: torch.Tensor,
    ref_pose: torch.Tensor,
    ref_velocity: torch.Tensor,
    aligned_pose: torch.Tensor,
    tracked: list[int],
    action: torch.Tensor,
    previous_action: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return unweighted body, head, and action tracking rewards.

    Args:
        q: Joint positions [rad].
        pose: Body positions [m] and xyzw orientations.
        velocity: World-frame body linear [m/s] and angular [rad/s] velocities.
        ref_q: Reference joint positions [rad].
        ref_pose: Reference body positions [m] and xyzw orientations.
        ref_velocity: Reference linear [m/s] and angular [rad/s] velocities.
        aligned_pose: Reference positions [m] and xyzw orientations aligned with the torso heading.
        tracked: Body indices used by the full-body rewards.
        action: Current normalized policy action.
        previous_action: Previous normalized policy action.

    Returns:
        Per-environment reward terms before weighting and time-step scaling.
    """
    pos_sq = (aligned_pose[:, tracked, :3] - pose[:, tracked, :3]).square().sum(-1)
    rot_sq = rotation_error(aligned_pose[:, tracked, 3:], pose[:, tracked, 3:]).square()
    lin_sq = (ref_velocity[:, tracked, :3] - velocity[:, tracked, :3]).square().sum(-1)
    ang_sq = (ref_velocity[:, tracked, 3:] - velocity[:, tracked, 3:]).square().sum(-1)
    root_pos_sq = (ref_pose[:, 19, :3] - pose[:, 19, :3]).square().sum(-1)
    root_ori_sq = rotation_error(ref_pose[:, 19, 3:], pose[:, 19, 3:]).square()
    # Head links occupy motion bodies 20:22, independently of critic body selection.
    head_pos_sq = (aligned_pose[:, 20:22, :3] - pose[:, 20:22, :3]).square().sum(-1)
    head_ori_sq = rotation_error(aligned_pose[:, 20:22, 3:], pose[:, 20:22, 3:]).square()
    terms = {
        "motion_global_anchor_pos": torch.exp(-root_pos_sq / 0.3**2),
        "motion_global_anchor_ori": torch.exp(-root_ori_sq / 0.4**2),
        "motion_body_pos": torch.exp(-pos_sq.mean(-1) / 0.3**2),
        "motion_body_ori": torch.exp(-rot_sq.mean(-1) / 0.4**2),
        "motion_body_lin_vel": torch.exp(-lin_sq.mean(-1)),
        "motion_body_ang_vel": torch.exp(-ang_sq.mean(-1) / 3.14**2),
        "action_rate_l2": (action - previous_action).square().sum(-1),
        "head_joint_pos": torch.exp(-(q[:, 19:21] - ref_q[:, 19:21]).square().mean(-1) / 0.2**2),
        "head_body_ori": torch.exp(-head_ori_sq.mean(-1) / 0.2**2),
        "head_body_pos": torch.exp(-head_pos_sq.mean(-1) / 0.1**2),
    }
    return terms


REWARD_WEIGHTS = {
    "motion_global_anchor_pos": 0.5,
    "motion_global_anchor_ori": 0.5,
    "motion_body_pos": 1.0,
    "motion_body_ori": 1.0,
    "motion_body_lin_vel": 1.0,
    "motion_body_ang_vel": 1.0,
    "action_rate_l2": -0.1,
    "joint_limit": -10.0,
    "undesired_contacts": -0.1,
    "head_joint_pos": 0.5,
    "head_body_ori": 0.5,
    "head_body_pos": 0.5,
}
