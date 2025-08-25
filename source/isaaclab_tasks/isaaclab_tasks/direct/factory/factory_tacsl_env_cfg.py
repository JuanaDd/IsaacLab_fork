# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg, ContactSensorCfg, TactileSensorCfg, TiledCameraCfg
from .factory_tasks_cfg import ASSET_DIR, FactoryTask, GearMesh, NutThread, PegInsert
from .factory_env_cfg import FactoryEnvCfg


@configclass
class FactoryTacSLEnvCfg(FactoryEnvCfg):
    decimation = 8
    action_space = 6
    # num_*: will be overwritten to correspond to obs_order, state_order.
    observation_space = 21
    state_space = 72
    obs_order: list = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel"]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
    ]

    episode_length_s = 10.0  # Probably need to override.
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,  # Important to avoid interpenetration.
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=1,  # Important for stable simulation.
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0, clone_in_fabric=False)

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/yvetted/project/test_asset/franka_mimic_tacsl.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint2": 0.00,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=87,
                velocity_limit_sim=124.6,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=12,
                velocity_limit_sim=149.5,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit_sim=40.0,
                velocity_limit_sim=0.04,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )

    tactile_sensor = TactileSensorCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_r15_left",
        # update_period=1 / 60,  # 60 Hz
        history_length=0,
        debug_vis=False,
        # Sensor configuration
        sensor_type="gelsight_r15",
        # enable_camera_tactile=args_cli.use_tactile_rgb,
        enable_camera_tactile=True,
        enable_force_field=False,
        # Elastomer configuration
        elastomer_link_name="elastomer",

        # elastomer_tip_link_name="elastomer_tip",
        elastomer_tip_link_name="panda_leftfingertip",
        # Force field configuration
        num_tactile_rows=20,
        num_tactile_cols=25,
        tactile_margin=0.003,
        sdf_tool="physx",
        # Indenter configuration (will be set based on indenter type)
        indenter_actor_name="indenter",  # Will be updated based on indenter type
        indenter_link_name="indenter",  # Will be updated based on indenter type
        # Force field physics parameters
        tactile_kn=1.0,
        tactile_damping=0.003,
        tactile_mu=2.0,
        tactile_kt=0.1,
        # Compliant dynamics
        compliance_stiffness=70.0,
        compliant_damping=1.0,
        use_acceleration_spring=False,
        # Camera configuration
        # camera_cfg= TiledCameraCfg(
        camera_cfg= CameraCfg(
            prim_path="/World/envs/env_.*/Robot/gelsight_r15_left/elastomer_tip/tactile_cam",
            update_period=1 / 60,  # 60 Hz
            height=320,
            width=240,
            # data_types=["rgb", "distance_to_image_plane"],
            data_types=["distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=0.020342857142857145 * 100,
                focus_distance=400.0 / 1000,
                horizontal_aperture=0.0119885 * 2 * 100,
                clipping_range=(0.0001, 1.0e5),
            ),
            # offset=TiledCameraCfg.OffsetCfg(
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, -0.020342857142857145 + 0.00175), rot=(0.5, 0.5, -0.5, 0.5), convention="world"
            ),
        ),
        # Visualization
        visualize_tactile_points=False,


    )

    debug_vis_folder = "/media/yvetted/juana_ssd/tacsl_isaaclab/"
    debug_vis_freq = 5


@configclass
class FactoryTacSLTaskPegInsertCfg(FactoryTacSLEnvCfg):
    task_name = "peg_insert"
    task = PegInsert()
    episode_length_s = 10.0
