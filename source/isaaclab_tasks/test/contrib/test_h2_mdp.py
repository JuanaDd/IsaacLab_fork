# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check head rewards with the original checkpoint's fourteen tracked bodies."""

import math

import torch

from isaaclab_tasks.contrib.h2_mimic import mdp
from isaaclab_tasks.contrib.h2_mimic.assets import MJ_BODIES, POLICY_JOINTS, TRACKED_BODIES


def test_head_rewards_use_head_links_without_extending_critic_tracking():
    """Penalize head errors independently of the final tracked wrist links."""
    heads = [MJ_BODIES.index(name) for name in ("head_pitch_link", "head_yaw_link")]
    head_joints = [POLICY_JOINTS.index(name) for name in ("head_pitch_joint", "head_yaw_joint")]
    wrists = [MJ_BODIES.index(name) for name in ("left_wrist_yaw_link", "right_wrist_yaw_link")]
    tracked = [MJ_BODIES.index(name) for name in TRACKED_BODIES]
    assert len(tracked) == 14 and not set(heads).intersection(tracked)
    pose = torch.zeros(2, len(MJ_BODIES), 7)
    pose[..., 6] = 1.0
    reference = pose.clone()
    joints = torch.zeros(2, len(POLICY_JOINTS))
    reference_joints = joints.clone()
    # Matching heads must receive full reward even when both wrists are wrong.
    pose[:, wrists, 0] = 1.0
    pose[:, wrists, 5:] = torch.tensor([math.sin(0.5), math.cos(0.5)])
    joints[1, head_joints] = torch.tensor([0.2, -0.4])
    pose[1, heads, :3] = torch.tensor([[0.06, 0.08, 0.0], [-0.12, 0.0, 0.16]])
    angles = torch.tensor([0.2, 0.4])
    pose[1, heads, 5] = torch.sin(angles / 2)
    pose[1, heads, 6] = torch.cos(angles / 2)
    velocity = torch.zeros(2, len(MJ_BODIES), 6)
    action = torch.zeros(2, 31)
    terms = mdp.tracking_terms(
        joints, pose, velocity, reference_joints, reference, velocity, reference, tracked, action, action
    )
    # The independently chosen errors give an exponent of -2.5 for all three terms.
    expected = torch.tensor([1.0, math.exp(-2.5)])
    for name in ("head_joint_pos", "head_body_pos", "head_body_ori"):
        torch.testing.assert_close(terms[name], expected)
        assert mdp.REWARD_WEIGHTS[name] == 0.5
    torch.testing.assert_close(terms["motion_body_pos"][0], terms["motion_body_pos"][1])
    torch.testing.assert_close(terms["motion_body_ori"][0], terms["motion_body_ori"][1])
