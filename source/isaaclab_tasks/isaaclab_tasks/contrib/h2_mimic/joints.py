# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint ordering shared by the H2 VBD task and native MJWarp policy playback.

This module has no simulator dependencies, allowing both Python environments to
use the same checkpoint convention without importing Isaac Lab or Newton.
"""

from collections.abc import Sequence

MJ_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "left_ankle_motor_joint",
    "left_knee_motor_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_motor_joint",
    "right_knee_motor_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_pitch_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "torso_constraint_R_joint",
    "torso_constraint_L_joint",
)

# Linkage outputs closed by the pushrods; everything else is actuated (31).
PASSIVE_JOINTS = (
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
# Original IsaacLab-mimic articulation order. The MJCF names ``ankle_motor``
# alias the old USD names ``ankle_A``; simulator storage remains name-resolved.
POLICY_JOINTS = (
    *MJ_JOINTS[:4],
    MJ_JOINTS[6],
    *MJ_JOINTS[4:6],
    *MJ_JOINTS[7:12],
    MJ_JOINTS[14],
    *MJ_JOINTS[12:14],
    *MJ_JOINTS[15:35],
    MJ_JOINTS[36],
    MJ_JOINTS[35],
)
POLICY_TO_MJ = tuple(MJ_JOINTS.index(name) for name in POLICY_JOINTS)
ACTUATED_JOINTS = tuple(j for j in POLICY_JOINTS if j not in PASSIVE_JOINTS)
assert len(ACTUATED_JOINTS) == 31


def policy_permutations(joint_names: Sequence[str], action_names: Sequence[str]) -> tuple[list[int], list[int]]:
    """Map native actor columns into VBD order and VBD actions into native order.

    Args:
        joint_names: Native ordering of the 37 observed joints.
        action_names: Native ordering of the 31 actuated joints.

    Returns:
        Column indices for the 194-dimensional actor observation and native actions.

    Raises:
        ValueError: If the target does not expose the checkpoint's joint and action sets.
    """
    if len(joint_names) != 37 or set(joint_names) != set(POLICY_JOINTS):
        raise ValueError("The target must expose the H2 checkpoint's 37 observed joints.")
    if len(action_names) != 31 or set(action_names) != set(ACTUATED_JOINTS):
        raise ValueError("The target must expose the H2 checkpoint's 31 actuated joints.")
    joints = [joint_names.index(name) for name in POLICY_JOINTS]
    previous_actions = [action_names.index(name) for name in ACTUATED_JOINTS]
    actor_columns = [
        *joints,
        *(37 + i for i in joints),
        *range(74, 89),
        *(89 + i for i in joints),
        *(126 + i for i in joints),
        *(163 + i for i in previous_actions),
    ]
    action_columns = [ACTUATED_JOINTS.index(name) for name in action_names]
    return actor_columns, action_columns
