# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example script demonstrating the TacSL tactile sensor implementation in IsaacLab.

This script shows how to use the TactileSensor for both camera-based and force field
tactile sensing with the gelsight finger setup.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tacsl/tacsl_example.py --enable_cameras --num_envs 2 --indenter_type nut

"""

import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import cv2
from PIL import Image

from isaaclab.app import AppLauncher
from isaaclab.utils.timer import Timer

# Add argparse arguments
parser = argparse.ArgumentParser(description="TacSL tactile sensor example.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--debug_mass", action="store_true", help="Debug mass.")
parser.add_argument("--save_viz", action="store_true", help="Visualize tactile data.")
parser.add_argument("--use_tactile_rgb", action="store_true", help="Use tactile RGB sensor data collection.")
parser.add_argument("--use_tactile_ff", action="store_true", help="Use tactile force field sensor data collection.")
parser.add_argument("--play_reset", action="store_true", help="Play reset animation.")
parser.add_argument(
    "--indenter_type", type=str, default="cube", choices=["cube", "sphere", "nut"], help="Type of indenter to use."
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# Import our TactileSensor
from isaaclab.sensors import CameraCfg, ContactSensorCfg, TactileSensorCfg, TiledCameraCfg
from isaaclab.sensors.tacsl_sensor.tactile_viz_utils import visualize_penetration_depth, visualize_tactile_shear_image
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"


@configclass
class TactileSensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with tactile sensors on the robot."""

    # Ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Robot with tactile sensor
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            # use local path for now
            usd_path="/home/yvetted/project/IsaacLab-Internal/scripts/tacsl/assets/gelsight_r15_finger.usd",
            # activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(math.sqrt(2) / 2, -math.sqrt(2) / 2, 0.0, 0.0),  # Exact 90° rotation
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )

    # Camera configuration for tactile sensing

    # TacSL Tactile Sensor
    tactile_sensor = TactileSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/elastomer",
        # update_period=1 / 60,  # 60 Hz
        history_length=0,
        debug_vis=False,
        # Sensor configuration
        sensor_type="gelsight_r15",
        enable_camera_tactile=args_cli.use_tactile_rgb,
        enable_force_field=args_cli.use_tactile_ff,
        # Elastomer configuration
        elastomer_link_name="elastomer",
        elastomer_tip_link_name="elastomer_tip",
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
        compliance_stiffness=200.0,
        compliant_damping=1.0,
        use_acceleration_spring=False,
        # Camera configuration
        camera_cfg= TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/elastomer_tip/tactile_cam",
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
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, 0.0, -0.020342857142857145 + 0.00175), rot=(0.5, 0.5, -0.5, 0.5), convention="world"
            ),
        ),
        # Visualization
        visualize_tactile_points=True,
    )
    # thumb_contact_sensor = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/elastomer",
    #     update_period=1.0 / 60.0,
    #     history_length=6,
    #     filter_prim_paths_expr=["{ENV_REGEX_NS}/indenter"],
    #     debug_vis=False,
    # )


@configclass
class CubeMeshTactileSceneCfg(TactileSensorsSceneCfg):
    """Scene with cube indenter."""

    # Cube indenter
    indenter = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/indenter",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.00327211),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0 + 0.06776, 0.51), rot=(1.0, 0.0, 0.0, 0.0)),
    )


@configclass
class CubeTactileSceneCfg(TactileSensorsSceneCfg):
    """Scene with cube indenter."""

    # Cube indenter
    indenter = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/indenter",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.00327211),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0 + 0.06776, 0.51), rot=(1.0, 0.0, 0.0, 0.0)),
    )


@configclass
class SphereTactileSceneCfg(TactileSensorsSceneCfg):
    """Scene with sphere indenter."""

    # Sphere indenter
    indenter = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/indenter",
        spawn=sim_utils.SphereCfg(
            radius=0.01,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.00327211),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0 + 0.06776, 0.52), rot=(1.0, 0.0, 0.0, 0.0)),
    )


@configclass
class NutArticulationTactileSceneCfg(TactileSensorsSceneCfg):
    """Scene with nut indenter."""

    # Nut indenter
    indenter = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/indenter",
        spawn=sim_utils.UrdfFileCfg(
            fix_base=True,
            merge_fixed_joints=False,
            make_instanceable=False,
            asset_path="/home/yvetted/project/IsaacGymEnvs/isaacgymenvs/tacsl_sensors/examples/assets/urdf/indenters/shape_sensing/factory_nut_m8_loose_subdiv_3x_6DOF.urdf",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0 + 0.06776, 0.498), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


@configclass
class NutTactileSceneCfg(TactileSensorsSceneCfg):
    """Scene with nut indenter."""

    # Nut indenter
    indenter = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/indenter",
        spawn=sim_utils.UsdFileCfg(
            # scale=(0.5, 0.5, 0.5),
            # use local path for now
            # activate_contact_sensors=True,
            usd_path="/home/yvetted/project/IsaacLab-Internal/scripts/tacsl/assets/factory_nut_m16.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.327211),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # fix_root_link=True,
                articulation_enabled=False
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0 + 0.06776, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def set_mass(scene: InteractiveScene):
    """Set mass of the indenter."""
    masses_per_env = [
        0.03,
        0.06,
        0.09,
        0.12,
        0.15,
        0.18,
        0.21,
        0.24,
        0.27,
        0.30,
        0.33,
        0.36,
        0.39,
        0.42,
        0.45,
        0.48,
        0.51,
        0.54,
        0.57,
        0.60,
    ][:num_envs]
    masses_per_env = [mass * 0.1 for mass in masses_per_env]
    mass_tensor = torch.tensor(masses_per_env, dtype=torch.float32, device="cpu")
    num_envs = scene.num_envs
    indices_tensor = torch.tensor(range(num_envs), dtype=torch.int32, device="cpu")
    print(f"Assigning masses per environment: {masses_per_env}")
    indenter = scene["indenter"]
    indenter.root_physx_view.set_masses(mass_tensor, indices=indices_tensor)
    print(f"Successfully set masses: {mass_tensor} on device: {mass_tensor.device}")


def run_simulator(sim, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Assign different masses to indenters in different environments
    num_envs = scene.num_envs

    # Define different masses for each environment
    if args_cli.debug_mass:
        set_mass(scene)

    # Create output directories for tactile data
    tactile_img_folder = "tactile_record"
    os.makedirs(tactile_img_folder, exist_ok=True)
    tactile_rgb_dir = os.path.join(tactile_img_folder, "tactile_rgb")
    os.makedirs(tactile_rgb_dir, exist_ok=True)
    tactile_depth_dir = os.path.join(tactile_img_folder, "tactile_depth")
    os.makedirs(tactile_depth_dir, exist_ok=True)
    tactile_shear_force_dir = os.path.join(tactile_img_folder, "tactile_shear_force")
    os.makedirs(tactile_shear_force_dir, exist_ok=True)
    tactile_taxim_dir = os.path.join(tactile_img_folder, "tactile_taxim")
    os.makedirs(tactile_taxim_dir, exist_ok=True)
    tactile_penetration_depth_dir = os.path.join(tactile_img_folder, "tactile_penetration_depth")
    os.makedirs(tactile_penetration_depth_dir, exist_ok=True)

    # Create constant downward force
    force_tensor = torch.zeros(scene.num_envs, 1, 3, device=sim.device)
    torque_tensor = torch.zeros(scene.num_envs, 1, 3, device=sim.device)

    force_tensor[:, 0, 2] = -3.0

    nrows = scene["tactile_sensor"].cfg.num_tactile_rows
    ncols = scene["tactile_sensor"].cfg.num_tactile_cols
    physics_timer = Timer()
    physics_total_time = 0.0
    physics_total_count = 0

    scene.update(sim_dt)
    scene.update(sim_dt)


    # Simulate physics
    b_play_reset = args_cli.play_reset
    b_benchmark = not b_play_reset

    while simulation_app.is_running() :
        # Reset every 200 steps
        if count > 120 and b_benchmark:
            print("Stopping benchmarking...")
            break
        if b_play_reset and count % 120 == 0:
            print(scene["tactile_sensor"].get_timing_summary())
            # break
            # Reset robot and indenter positions
            count = 0
            for entity in ["robot", "indenter"]:
                root_state = scene[entity].data.default_root_state.clone()
                print("root_state: ", entity, root_state)
                root_state[:, :3] += scene.env_origins
                scene[entity].write_root_state_to_sim(root_state)

            # break
            scene.reset()
            print("[INFO]: Resetting robot and indenter state...")

            # Reapply force after reset

        # Apply forces to simulation
        if count > 20:
            torque_tensor[:, 0, 2] = 0.05 / 2  # rotation
        scene["indenter"].set_external_force_and_torque(force_tensor, torque_tensor)

        # Step simulation
        scene.write_data_to_sim()
        physics_timer.start()
        sim.step()
        physics_timer.stop()
        physics_total_time += physics_timer.total_run_time
        physics_total_count += 1
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        # Access tactile sensor data
        tactile_data = scene["tactile_sensor"].data

        # Print tactile information every 30 steps
        if args_cli.save_viz:
            print(f"[INFO] Step {count}: Tactile sensor data available")

            if tactile_data.penetration_depth is not None:
                penetration_depth = tactile_data.penetration_depth.view((num_envs, nrows, ncols))
                penetration_depth_img_upsampled = visualize_penetration_depth(
                    penetration_depth[0].detach().cpu().numpy(), resolution=5, depth_multiplier=300.0
                )

                cv2.imwrite(
                    os.path.join(tactile_penetration_depth_dir, f"{count}.png"), penetration_depth_img_upsampled
                )
            if tactile_data.tactile_shear_force is not None:
                # visualize tactile forces
                tactile_normal_force = tactile_data.tactile_normal_force.view((num_envs, nrows, ncols))
                tactile_shear_force = tactile_data.tactile_shear_force.view((num_envs, nrows, ncols, 2))

                tactile_image = visualize_tactile_shear_image(
                    tactile_normal_force[0].detach().cpu().numpy(), tactile_shear_force[0].detach().cpu().numpy()
                )
                cv2.imwrite(os.path.join(tactile_shear_force_dir, f"{count}.png"), tactile_image * 255)

            if tactile_data.taxim_tactile is not None:
                print(f"  Tactile taxim shape: {tactile_data.taxim_tactile.shape}")
                taxim_data = tactile_data.taxim_tactile.cpu().numpy()
                taxim_tiled = np.concatenate(taxim_data, axis=0)
                cv2.imwrite(os.path.join(tactile_taxim_dir, f"{count}.png"), taxim_tiled)

            if tactile_data.tactile_rgb is not None:
                print(f"  Tactile RGB shape: {tactile_data.tactile_rgb.shape}")
                rgb_data = tactile_data.tactile_rgb.cpu().numpy()
                rgb_tiled = np.concatenate(rgb_data, axis=0)
                plt.imsave(os.path.join(tactile_rgb_dir, f"{count}.png"), rgb_tiled)

            if tactile_data.tactile_depth is not None and False:
                print(f"  Tactile depth shape: {tactile_data.tactile_depth.shape}")  # 1, 320, 240, 1
                depth_data = tactile_data.tactile_depth.cpu().numpy()
                depth_tiled = np.concatenate(depth_data, axis=0)
                # Save as float32 TIFF (preserves float values)
                depth_float = depth_tiled.squeeze().astype(np.float32)
                img = Image.fromarray(depth_float, mode="F")  # 'F' = 32-bit float
                img.save(os.path.join(tactile_depth_dir, f"{count}.tiff"))

            if tactile_data.tactile_points_pos_w is not None:
                print(f"  Tactile points shape: {tactile_data.tactile_points_pos_w.shape}")

            if tactile_data.tactile_normal_force is not None:
                print(f"  Tactile normal force shape: {tactile_data.tactile_normal_force.shape}")
                total_force = tactile_data.tactile_normal_force.sum().item()
                print(f"  Total normal force: {total_force:.4f}")

            # Print applied external force info
            print(f"  Applied external force: {force_tensor[0, 0, 2].item():.2f}N (downward)")

            
    print("--------------------------------")
    # Get timing summary from sensor and add physics timing
    timing_summary = scene["tactile_sensor"].get_timing_summary()
    
    # Add physics timing to the summary
    physics_avg = physics_total_time / (physics_total_count * scene.num_envs) if physics_total_count > 0 else 0.0
    timing_summary["physics_total"] = physics_total_time
    timing_summary["physics_average"] = physics_avg
    timing_summary["physics_fps"] = 1 / physics_avg if physics_avg > 0 else 0.0
    
    print(timing_summary)



def main():
    """Main function."""
    # Initialize simulation
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005, 
        device=args_cli.device,
        physx=sim_utils.PhysxCfg(
            gpu_collision_stack_size=2**30,  # Important to prevent collisionStackSize buffer overflow in contact-rich environments.
        )
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set main camera using sim_utils function
    sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.0])

    # Create scene based on indenter type
    if args_cli.indenter_type == "cube":
        scene_cfg = CubeMeshTactileSceneCfg(num_envs=args_cli.num_envs, env_spacing=1.0)
        # scene_cfg = CubeTactileSceneCfg(num_envs=args_cli.num_envs, env_spacing=1.0)
        # Update tactile sensor configuration for cube
        scene_cfg.tactile_sensor.indenter_actor_name = "indenter"
        scene_cfg.tactile_sensor.indenter_link_name = "geometry"
    elif args_cli.indenter_type == "sphere":
        scene_cfg = SphereTactileSceneCfg(num_envs=args_cli.num_envs, env_spacing=1.0)
        # Update tactile sensor configuration for sphere
        scene_cfg.tactile_sensor.indenter_actor_name = "indenter"
        scene_cfg.tactile_sensor.indenter_link_name = "indenter"
    elif args_cli.indenter_type == "nut":
        scene_cfg = NutTactileSceneCfg(num_envs=args_cli.num_envs, env_spacing=1.0)
        # Update tactile sensor configuration for nut
        scene_cfg.tactile_sensor.indenter_actor_name = "indenter"
        scene_cfg.tactile_sensor.indenter_link_name = "factory_nut_loose"

    scene = InteractiveScene(scene_cfg)


    scene["tactile_sensor"].setup_compliant_materials()

    # Initialize simulation
    sim.reset()
    print("[INFO]: Setup complete...")
    # print(f"[INFO]: Tactile sensor: {scene['tactile_sensor']}")
    scene["tactile_sensor"].get_initial_render()

    # Run simulation
    run_simulator(sim, scene)


if __name__ == "__main__":
    # Run the main function
    main()
    print("--------------------------------")
    # Close sim app
    simulation_app.close()
    
