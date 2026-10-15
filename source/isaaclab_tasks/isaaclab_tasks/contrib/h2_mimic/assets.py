# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Closed-loop H2 asset helpers adapted from newton_f/sim_h2_loop.py.

The six rods retain the source implementation\'s geometry, mass and inertia.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET

import newton
import numpy as np
import warp as wp

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

# 38 bodies in MuJoCo order (== npz body axis == Newton MJCF DFS order).
MJ_BODIES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "left_ankle_pitch_link",
    "left_ankle_A_link",
    "left_knee_motor_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "right_ankle_pitch_link",
    "right_ankle_A_link",
    "right_knee_motor_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "head_pitch_link",
    "head_yaw_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "torso_constraint_R_link",
    "torso_constraint_L_link",
)
TRACKED_BODIES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)

# Original IsaacLab-mimic: critical nominal damping, plus 0.05 passive damping
# added to the implicit drive in build_h2. Action scale uses the nominal effort.
_NATURAL_FREQ = 2.0 * math.pi * 10.0
_DAMPING_RATIO = 1.0
PASSIVE_DAMPING = 0.05
SOFT_JOINT_LIMIT_FACTOR = 0.98
_ACTUATOR_GROUPS = (
    # (regex, armature, effort limit)
    (r".*_hip_(pitch|roll|yaw)_joint|.*_knee_motor_joint", 0.160478, 254.0),
    (r".*_ankle_(roll|motor)_joint", 0.0190185, 53.7),
    (r"waist_yaw_joint|.*_shoulder_.*_joint|.*_elbow_joint|torso_constraint_[LR]_joint", 0.0251019, 111.0),
    (r"head_.*_joint", 0.01, 50.0),
    (r".*_wrist_roll_joint", 0.01, 60.0),
    (r".*_wrist_(pitch|yaw)_joint", 0.01, 10.0),
)
DEFAULT_JOINT_POS = {
    "left_shoulder_pitch_joint": 0.297,
    "right_shoulder_pitch_joint": 0.297,
    "left_shoulder_roll_joint": 0.281,
    "right_shoulder_roll_joint": -0.281,
    "left_elbow_joint": 0.718,
    "right_elbow_joint": 0.718,
}


def actuator_params(name: str) -> tuple[float, float, float, float]:
    """Return proportional [Nm/rad] and active damping [Nm*s/rad] gains, effort [Nm], and action scale [rad]."""
    for pattern, armature, effort in _ACTUATOR_GROUPS:
        if re.fullmatch(pattern, name):
            kp = armature * _NATURAL_FREQ**2
            kd = 2.0 * _DAMPING_RATIO * armature * _NATURAL_FREQ
            return kp, kd, effort, 0.25 * effort / kp
    raise KeyError(name)


_MIN_BODY_MASS = 0.02
_MIN_BODY_INERTIA = 2.0e-6

_ROD_COLOR = wp.vec3(0.80, 0.68, 0.28)
_ROD_RADIUS = 0.008
_ROD_LINEAR_DENSITY = 0.3  # kg/m for reconstructed pushrods

_LOOPS_MJCF = (
    (
        "left_ankle_pitch_link",
        "left_ankle_output_rod_site",
        "left_ankle_A_link",
        "left_ankle_motor_rod_site",
        "left_ankle_rod",
    ),
    (
        "right_ankle_pitch_link",
        "right_ankle_output_rod_site",
        "right_ankle_A_link",
        "right_ankle_motor_rod_site",
        "right_ankle_rod",
    ),
    (
        "left_knee_link",
        "left_knee_output_rod_site",
        "left_knee_motor_link",
        "left_knee_motor_rod_site",
        "left_knee_rod",
    ),
    (
        "right_knee_link",
        "right_knee_output_rod_site",
        "right_knee_motor_link",
        "right_knee_motor_rod_site",
        "right_knee_rod",
    ),
    (
        "waist_yaw_link",
        "left_waist_output_rod_site",
        "torso_constraint_L_link",
        "left_waist_motor_rod_site",
        "waist_rod_L",
    ),
    (
        "waist_yaw_link",
        "right_waist_output_rod_site",
        "torso_constraint_R_link",
        "right_waist_motor_rod_site",
        "waist_rod_R",
    ),
)


def _find(labels: list[str], suffix: str) -> int:
    for i, label in enumerate(labels):
        if label.rsplit("/", 1)[-1] == suffix:
            return i
    raise KeyError(f"Missing '{suffix}' in {[label.rsplit('/', 1)[-1] for label in labels]}")


def _site_world_pos(builder: newton.ModelBuilder, site_label: str) -> np.ndarray:
    """World position of an imported MJCF site.

    ``add_mjcf`` turns each ``<site>`` into a non-colliding shape on its parent
    body, so the anchor is the body pose composed with the shape's local xform.
    """
    for si, label in enumerate(builder.shape_label):
        if label.rsplit("/", 1)[-1] != site_label:
            continue
        body = builder.shape_body[si]
        x_b = wp.transform_identity() if body < 0 else wp.transform(*builder.body_q[body])
        return np.array(x_b * wp.transform(*builder.shape_transform[si]))[:3]
    raise KeyError(f"Missing site '{site_label}' among the imported shapes")


def _rest_pose_fk(builder: newton.ModelBuilder) -> None:
    """Initialize q=0 body poses from tree joint frames before closing the loops."""
    for j in range(len(builder.joint_type)):
        parent = builder.joint_parent[j]
        child = builder.joint_child[j]
        x_p = wp.transform(*builder.joint_X_p[j])
        x_c = wp.transform(*builder.joint_X_c[j])
        t_parent = wp.transform_identity() if parent < 0 else wp.transform(*builder.body_q[parent])
        builder.body_q[child] = t_parent * x_p * wp.transform_inverse(x_c)


def _single_closed_loop_articulation(builder: newton.ModelBuilder, label: str) -> None:
    """Merge every joint (including the loop-closing balls) into one articulation.

    The VBD sparse articulation solver assembles the whole joint graph of an
    articulation as a block-sparse system, so loop-closing joints must live in
    the same articulation as the spanning tree.
    """
    joint_count = len(builder.joint_articulation)
    for joint_index in range(joint_count):
        builder.joint_articulation[joint_index] = 0
    builder.articulation_start = [0]
    builder.articulation_end = [joint_count]
    builder.articulation_label = [label]
    builder.articulation_world = [builder.current_world]


def _local_point(body_xform: wp.transform, world_p: np.ndarray) -> wp.vec3:
    inv = wp.transform_inverse(body_xform)
    return wp.transform_point(inv, wp.vec3(float(world_p[0]), float(world_p[1]), float(world_p[2])))


def _box_inertia(mass: float, hx: float, hy: float, hz: float) -> wp.mat33:
    x, y, z = 2.0 * hx, 2.0 * hy, 2.0 * hz
    return wp.mat33(
        (mass * (y * y + z * z) / 12.0, 0.0, 0.0),
        (0.0, mass * (x * x + z * z) / 12.0, 0.0),
        (0.0, 0.0, mass * (x * x + y * y) / 12.0),
    )


def _quat_between(a: np.ndarray, b: np.ndarray) -> wp.quat:
    a = a.astype(np.float64) / np.linalg.norm(a)
    b = b.astype(np.float64) / np.linalg.norm(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot < -0.999999:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1.0e-8:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis)
        return wp.quat_from_axis_angle(wp.vec3(*axis.astype(np.float32)), math.pi)
    cross = np.cross(a, b)
    scale = math.sqrt((1.0 + dot) * 2.0)
    inv = 1.0 / scale
    return wp.quat(float(cross[0] * inv), float(cross[1] * inv), float(cross[2] * inv), float(0.5 * scale))


def _add_rod_body(builder: newton.ModelBuilder, center: np.ndarray, rod_q: wp.quat, half_len: float, label: str) -> int:
    mass = max(_MIN_BODY_MASS, _ROD_LINEAR_DENSITY * 2.0 * half_len)
    rod = builder.add_link(
        xform=wp.transform(wp.vec3(*center.astype(np.float32)), rod_q),
        com=wp.vec3(0.0, 0.0, 0.0),
        inertia=_box_inertia(mass, _ROD_RADIUS, _ROD_RADIUS, half_len),
        mass=mass,
        label=label,
        lock_inertia=True,
    )
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = 0.0
    cfg.has_shape_collision = False
    cfg.has_particle_collision = False
    builder.add_shape_capsule(
        rod, radius=_ROD_RADIUS, half_height=half_len, cfg=cfg, color=_ROD_COLOR, label=f"{label}_bar"
    )
    return rod


def _end_frames(
    builder: newton.ModelBuilder, b_link: int, rod: int, p_anchor: np.ndarray, world_rot: wp.quat
) -> tuple[wp.transform, wp.transform]:
    """Parent/child joint frames at `p_anchor` whose world rotation is `world_rot`."""
    t_link = wp.transform(*builder.body_q[b_link])
    t_rod = wp.transform(*builder.body_q[rod])
    q_link = wp.transform_get_rotation(t_link)
    q_rod = wp.transform_get_rotation(t_rod)
    px = wp.transform(_local_point(t_link, p_anchor), wp.quat_inverse(q_link) * world_rot)
    cx = wp.transform(_local_point(t_rod, p_anchor), wp.quat_inverse(q_rod) * world_rot)
    return px, cx


def _add_pushrod(
    builder: newton.ModelBuilder,
    b_a: int,
    p_a: np.ndarray,
    b_b: int,
    p_b: np.ndarray,
    label: str,
) -> None:
    """Reconstruct a ball-ended rigid rod between two attachment sites."""
    rod_vec = p_b - p_a
    rod_len = float(np.linalg.norm(rod_vec))
    rod_q = _quat_between(np.array([0.0, 0.0, 1.0], dtype=np.float32), rod_vec)

    rod = _add_rod_body(builder, 0.5 * (p_a + p_b), rod_q, 0.5 * rod_len, label)

    for b_link, p_anchor, end in ((b_a, p_a, "a"), (b_b, p_b, "b")):
        px, cx = _end_frames(builder, b_link, rod, p_anchor, wp.quat_identity())
        builder.add_joint_ball(parent=b_link, child=rod, parent_xform=px, child_xform=cx, label=f"{label}_ball_{end}")


def build_h2(mjcf_path: str, *, parse_visuals: bool = False) -> newton.ModelBuilder:
    """Build six ball-ended rods with the original IsaacLab-mimic PD gains.

    Args:
        mjcf_path: Path to the H2 MJCF, including its six pairs of pushrod sites.
        parse_visuals: Whether to include visual meshes and visible collision-only links.

    Returns:
        A 44-body builder containing the H2 and six rigid pushrods.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_mjcf(
        mjcf_path,
        floating=True,
        enable_self_collisions=False,
        parse_sites=True,
        skip_equality_constraints=True,
        parse_visuals=parse_visuals,
    )
    _rest_pose_fk(builder)
    base = ET.parse(mjcf_path).getroot().find("./worldbody/body")
    base_pos = [float(v) for v in base.get("pos", "0 0 0").split()]
    for index, transform in enumerate(builder.body_q):
        pose = list(transform)
        pose[:3] = np.asarray(pose[:3]) + base_pos
        builder.body_q[index] = pose
        if builder.body_mass[index] < _MIN_BODY_MASS:
            builder.body_mass[index] = _MIN_BODY_MASS
            builder.body_inv_mass[index] = 1.0 / _MIN_BODY_MASS
            inertia = np.array(builder.body_inertia[index]).reshape(3, 3)
            inertia[np.diag_indices(3)] = np.maximum(np.diag(inertia), _MIN_BODY_INERTIA)
            builder.body_inertia[index] = wp.mat33(inertia)
            builder.body_inv_inertia[index] = wp.mat33(np.linalg.inv(inertia))
    static_shapes = {i for i, body in enumerate(builder.shape_body) if body < 0}
    builder.shape_collision_filter_pairs = type(builder.shape_collision_filter_pairs)(
        pair
        for pair in builder.shape_collision_filter_pairs
        if pair[0] not in static_shapes and pair[1] not in static_shapes
    )
    for link_a, site_a, link_b, site_b, label in _LOOPS_MJCF:
        _add_pushrod(
            builder,
            _find(builder.body_label, link_a),
            _site_world_pos(builder, site_a),
            _find(builder.body_label, link_b),
            _site_world_pos(builder, site_b),
            label,
        )
    _single_closed_loop_articulation(builder, "h2_closeloop")
    for j, label in enumerate(builder.joint_label):
        name = label.rsplit("/", 1)[-1]
        if name not in MJ_JOINTS:
            continue
        dof = builder.joint_qd_start[j]
        if name in ("torso_constraint_L_joint", "torso_constraint_R_joint"):
            builder.joint_limit_lower[dof] = -1.0
            builder.joint_limit_upper[dof] = 1.3
        builder.joint_target_ke[dof] = 0.0
        builder.joint_target_kd[dof] = PASSIVE_DAMPING
        builder.joint_armature[dof] = 0.01
        builder.joint_effort_limit[dof] = 360.0
        if name in ACTUATED_JOINTS:
            kp, kd, effort, _ = actuator_params(name)
            builder.joint_armature[dof] = kp / _NATURAL_FREQ**2
            builder.joint_target_ke[dof] = kp
            builder.joint_target_kd[dof] = kd + PASSIVE_DAMPING
            builder.joint_effort_limit[dof] = effort
            builder.joint_target_mode[dof] = newton.JointTargetMode.POSITION_VELOCITY
    for i in range(len(builder.shape_material_mu)):
        builder.shape_material_mu[i] = 0.6
    if parse_visuals:
        visible_bodies = {
            body
            for body, flags in zip(builder.shape_body, builder.shape_flags, strict=True)
            if flags & int(newton.ShapeFlags.VISIBLE)
        }
        for i, body in enumerate(builder.shape_body):
            if body < 0:
                builder.shape_color[i] = (0.18, 0.21, 0.26)
            # Some H2 links provide a collision mesh without a visual mesh.
            if body not in visible_bodies and builder.shape_flags[i] & int(newton.ShapeFlags.COLLIDE_SHAPES):
                builder.shape_flags[i] |= int(newton.ShapeFlags.VISIBLE)
    return builder
