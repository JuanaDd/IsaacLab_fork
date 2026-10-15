# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native Isaac Lab manager for the local Newton sparse VBD H2 solver."""

from __future__ import annotations

from typing import TYPE_CHECKING

import newton
import numpy as np
import warp as wp
from isaaclab_newton.physics.newton_manager import NewtonManager
from isaaclab_newton.physics.vbd_manager import NewtonVBDManager

from .assets import ACTUATED_JOINTS, DEFAULT_JOINT_POS, actuator_params

if TYPE_CHECKING:
    from .configs import H2VBDSolverCfg


@wp.kernel
def project_pd_target(
    coord: wp.array[int],
    target_index: wp.array[int],
    kp: wp.array[float],
    effort: wp.array[float],
    desired: wp.array[float],
    q: wp.array[float],
    target: wp.array[float],
):
    """Limit the proportional torque [Nm], preserving the original policy's PD semantics."""
    i = wp.tid()
    center = q[coord[i]]
    radius = effort[i] / kp[i]
    target[target_index[i]] = wp.clamp(desired[i], center - radius, center + radius)


@wp.kernel
def snapshot_contact_history(
    world: wp.array[int],
    rebaseline: wp.array[wp.bool],
    current: wp.array[wp.transform],
    previous: wp.array[wp.transform],
    snapshot: wp.array[wp.transform],
):
    i = wp.tid()
    slot = world[i]
    if slot < 0:
        slot = rebaseline.shape[0] - 1
    snapshot[i] = previous[i]
    if rebaseline[slot]:
        snapshot[i] = current[i]


@wp.kernel
def sum_contact_forces(
    count: wp.array[int],
    body0: wp.array[int],
    body1: wp.array[int],
    force_on_body1: wp.array[wp.vec3],
    net_force: wp.array[wp.vec3],
):
    i = wp.tid()
    if i < count[0]:
        if body0[i] >= 0:
            wp.atomic_add(net_force, body0[i], -force_on_body1[i])
        if body1[i] >= 0:
            wp.atomic_add(net_force, body1[i], force_on_body1[i])


@wp.kernel
def update_contact_history(net_force: wp.array[wp.vec3], history: wp.array2d[wp.vec3]):
    i = wp.tid()
    history[i, 2] = history[i, 1]
    history[i, 1] = history[i, 0]
    history[i, 0] = net_force[i]


class NewtonH2VBDManager(NewtonVBDManager):
    """Preserve closed-loop maximal coordinates and update PD each substep."""

    @classmethod
    def _create_solver(cls, model: newton.Model, solver_cfg: H2VBDSolverCfg) -> newton.solvers.SolverVBD:
        # VBD allocates persistent contact history using the pipeline capacity.
        # It must exist before the base manager captures its simulation graph.
        cls._initialize_contacts()
        labels = [label.rsplit("/", 1)[-1] for label in model.joint_label]
        joints = [j for j, name in enumerate(labels) if name in ACTUATED_JOINTS]
        worlds = model.joint_world.numpy()
        action_order = {name: index for index, name in enumerate(ACTUATED_JOINTS)}
        joints.sort(key=lambda joint: (int(worlds[joint]), action_order[labels[joint]]))
        params = np.asarray([actuator_params(labels[j]) for j in joints], dtype=np.float32)
        cls.pd_coord = wp.array(model.joint_q_start.numpy()[joints], dtype=int, device=model.device)
        cls.pd_index = wp.array(model.joint_target_q_start.numpy()[joints], dtype=int, device=model.device)
        cls.pd_kp, cls.pd_kd, cls.pd_effort = [
            wp.array(params[:, i], dtype=float, device=model.device) for i in range(3)
        ]
        cls.desired = wp.array(
            [DEFAULT_JOINT_POS.get(labels[j], 0.0) for j in joints], dtype=float, device=model.device
        )
        solver = newton.solvers.SolverVBD(
            model,
            iterations=solver_cfg.iterations,
            rigid_compliant_alm=False,
            rigid_articulation_solve="block_sparse_joints",
            rigid_joint_armature=True,
            rigid_articulation_level_parallel=True,
            rigid_articulation_relaxation=solver_cfg.relaxation,
            rigid_articulation_diagonal_regularization=0.0,
            rigid_avbd_alpha=0.0,
            rigid_avbd_beta=0.0,
            rigid_joint_linear_ke=5.0e6,
            rigid_joint_angular_ke=5.0e6,
            rigid_joint_linear_kd=125.0,
            rigid_joint_angular_kd=125.0,
            rigid_contact_history=True,
            rigid_body_contact_buffer_size=256,
        )
        cls.contact_previous_pose = wp.clone(model.body_q)
        cls.contact_net_force = wp.zeros(model.body_count, dtype=wp.vec3, device=model.device)
        cls.contact_force_history = wp.zeros((model.body_count, 3), dtype=wp.vec3, device=model.device)
        # Allocate force-reporting scratch before CUDA graph capture. The real
        # substep dt is supplied at every sample below.
        solver.collect_rigid_contact_forces(model.body_q, model.body_q, cls._contacts, 0.005)
        return solver

    @classmethod
    def _initialize_contacts(cls) -> None:
        if cls._collision_pipeline is not None:
            return
        NewtonManager._collision_pipeline = newton.CollisionPipeline(
            cls._model,
            broad_phase="nxn",
            contact_matching="sticky",
            contact_matching_pos_threshold=0.005,
            rigid_contact_max=128 * cls._model.world_count,
        )
        NewtonManager._contacts = cls._collision_pipeline.contacts()

    @classmethod
    def _reset_solver_internals(cls, world_mask: wp.array | None) -> None:
        super()._reset_solver_internals(world_mask)
        cls._collision_pipeline.reset_contact_matching(world_mask=world_mask)

    @classmethod
    def _eval_fk_impl(cls, world_reset_mask: wp.array | None, fk_mask: wp.array | None) -> None:
        # Generic FK traverses a tree and would overwrite the six rod poses.
        # The environment writes complete body poses and COM twists on reset.
        for state in (cls._state_0, cls._state_1):
            newton.eval_ik(cls._model, state, state.joint_q, state.joint_qd)

    @classmethod
    def _step_solver(
        cls,
        state_0: newton.State,
        state_1: newton.State,
        control: newton.Control,
        contacts: newton.Contacts | None,
        substep_dt: float,
    ) -> None:
        newton.eval_ik(cls._model, state_0, state_0.joint_q, state_0.joint_qd)
        wp.launch(
            project_pd_target,
            dim=cls.desired.shape[0],
            inputs=[
                cls.pd_coord,
                cls.pd_index,
                cls.pd_kp,
                cls.pd_effort,
                cls.desired,
                state_0.joint_q,
            ],
            outputs=[control.joint_target_q],
            device=cls._model.device,
        )
        # VBD advances body_q_prev in step(); force reporting needs its input
        # value, including the first-step/reset rebaseline performed by VBD.
        wp.launch(
            snapshot_contact_history,
            dim=cls._model.body_count,
            inputs=[
                cls._model.body_world,
                cls._solver._rigid_pose_rebaseline_mask,
                state_0.body_q,
                cls._solver.body_q_prev,
            ],
            outputs=[cls.contact_previous_pose],
            device=cls._model.device,
        )
        super()._step_solver(state_0, state_1, control, contacts, substep_dt)
        body0, body1, _, _, force, count = cls._solver.collect_rigid_contact_forces(
            state_1.body_q, cls.contact_previous_pose, contacts, substep_dt
        )
        cls.contact_net_force.zero_()
        wp.launch(
            sum_contact_forces,
            dim=force.shape[0],
            inputs=[count, body0, body1, force],
            outputs=[cls.contact_net_force],
            device=cls._model.device,
        )
        wp.launch(
            update_contact_history,
            dim=cls._model.body_count,
            inputs=[cls.contact_net_force],
            outputs=[cls.contact_force_history],
            device=cls._model.device,
        )
        newton.eval_ik(cls._model, state_1, state_1.joint_q, state_1.joint_qd)


# Backend discovery requires the concrete class name to start with Newton.
H2VBDManager = NewtonH2VBDManager
