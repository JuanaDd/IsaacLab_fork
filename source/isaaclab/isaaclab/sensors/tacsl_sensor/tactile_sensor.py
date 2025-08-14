# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tactile sensor implementation for IsaacLab."""

from __future__ import annotations

import itertools
import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from isaaclab.utils.timer import Timer

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import isaacsim.core.utils.torch as torch_utils
import omni.log
from isaacsim.core.prims import SdfShapePrim
from isaacsim.core.simulation_manager import SimulationManager
from omni.physx.scripts import physicsUtils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.materials.physics_materials import spawn_rigid_body_material
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

from ..camera import Camera, TiledCamera
from ..sensor_base import SensorBase
from .tactile_sensor_data import TactileSensorData
from .tactile_utils import gelsightRender

if TYPE_CHECKING:
    from .tactile_sensor_cfg import TactileSensorCfg

# Try importing optional dependencies
try:
    import trimesh

    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    trimesh = None

try:
    import urdfpy

    URDFPY_AVAILABLE = True
except ImportError:
    URDFPY_AVAILABLE = False
    urdfpy = None

try:
    from pysdf import SDF

    PYSDF_AVAILABLE = True
except ImportError:
    PYSDF_AVAILABLE = False
    SDF = None

from scipy.spatial.transform import Rotation as R


class TactileSensor(SensorBase):
    """A tactile sensor for both camera-based and force field tactile sensing.

    This sensor provides:
    1. Camera-based tactile sensing: RGB and depth images from tactile surface
    2. Force field tactile sensing: Penalty-based normal and shear forces using SDF queries

    The sensor can be configured to use either or both sensing modalities.
    It follows the same initialization and update patterns as other IsaacLab sensors.
    """

    cfg: TactileSensorCfg
    """The configuration parameters."""

    def __init__(self, cfg: TactileSensorCfg):
        """Initializes the tactile sensor object.

        Args:
            cfg: The configuration parameters.
        """
        # Initialize instance variables before calling super().__init__
        # This is needed because super().__init__ may call _set_debug_vis_impl

        # Create empty variables for storing output data
        self._data: TactileSensorData = TactileSensorData()

        # Camera-based tactile sensing
        self._camera_sensor: Camera | TiledCamera | None = None
        self._nominal_tactile: dict | None = None

        # Force field tactile sensing
        self._tactile_pos_local: torch.Tensor | None = None
        self._tactile_quat_local: torch.Tensor | None = None
        self._sdf_object: Any | None = None
        self._indenter_mesh: Any | None = None
        self._indenter_mesh_local_tf: tuple[torch.Tensor, torch.Tensor] | None = None

        # Physics views
        self._physics_sim_view = None
        self._elastomer_body_view = None
        self._elastomer_tip_view = None
        self._indenter_body_view = None

        # SDF querying (for PhysX)
        self._sdf_shape_global_ids_per_env: torch.Tensor | None = None

        # Internal state
        self._elastomer_link_id: int | None = None
        self._indenter_link_id: int | None = None

        # Visualization
        self._tactile_visualizer: VisualizationMarkers | None = None

        # Timing tracking attributes
        self._camera_timing_total: float = 0.0
        self._force_field_timing_total: float = 0.0
        self._timing_call_count: int = 0
        self._camera_timer: Timer = Timer()
        self._force_field_timer: Timer = Timer()

        # Now call parent class constructor
        super().__init__(cfg)

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Tactile sensor @ '{self.cfg.prim_path}': \n"
            f"\tsensor type       : {self.cfg.sensor_type}\n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tcamera enabled    : {self.cfg.enable_camera_tactile}\n"
            f"\tforce field enabled: {self.cfg.enable_force_field}\n"
            f"\tnum instances     : {self.num_instances}\n"
        )

    """
    Properties
    """

    @property
    def num_instances(self) -> int:
        return self._num_envs

    @property
    def data(self) -> TactileSensorData:
        # Update sensors if needed
        self._update_outdated_buffers()
        # Return the data
        return self._data

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets the sensor internals."""
        # Reset camera sensor if enabled
        self.reset_timing_statistics()
        if self._camera_sensor is not None:
            self._camera_sensor.reset(env_ids)

    """
    Implementation
    """

    def _initialize_impl(self):
        """Initializes the sensor-related handles and internal buffers."""
        super()._initialize_impl()

        # Obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()

        # Initialize camera-based tactile sensing
        if self.cfg.enable_camera_tactile and self.cfg.camera_cfg is not None:
            self._initialize_camera_tactile()

        # Initialize force field tactile sensing
        if self.cfg.enable_force_field:
            self._initialize_force_field()

        # Initialize visualization
        if self.cfg.debug_vis and self.cfg.visualize_tactile_points:
            self._initialize_visualization()

        # Setup compliant contact materials

    def setup_compliant_materials(self):
        """Setup compliant contact materials for the elastomer."""
        # Get the current stage
        if self.cfg.enable_compliant_contact:
            self._setup_compliant_materials()

    def get_initial_render(self):
        if not self.cfg.enable_camera_tactile:
            return None
        if not self._camera_sensor.is_initialized:
            assert self._camera_sensor.is_initialized, "Camera sensor is not initialized"

        self._camera_sensor.update(dt=0.0)

        # get the initial render
        initial_render = self._camera_sensor.data.output
        if initial_render is None:
            assert False, "Initial render is None"

        if self._nominal_tactile is not None:
            assert False, "Nominal tactile is not None"

        self._nominal_tactile = dict()
        for key, value in initial_render.items():
            print(f"key: {key}, value: {value.shape}")
            self._nominal_tactile[key] = value.clone()

        return self._nominal_tactile

    def _initialize_camera_tactile(self):
        """Initialize camera-based tactile sensing."""
        if self.cfg.camera_cfg is None:
            omni.log.warn("Camera configuration is None. Disabling camera-based tactile sensing.")
            return

        # gelsightRender
        self.taxim_gelsight = gelsightRender("gelsight_r15", device=self.device)

        # Create camera sensor
        self._camera_sensor = TiledCamera(self.cfg.camera_cfg)

        # Initialize camera if not already done
        # TODO: Juana: this is a hack to initialize the camera sensor. Should camera_sensor be managed by TacSL sensor or InteractiveScene?
        if not self._camera_sensor.is_initialized:
            self._camera_sensor._initialize_impl()
            self._camera_sensor._is_initialized = True

        omni.log.info("Camera-based tactile sensing initialized.")

    def _initialize_force_field(self):
        """Initialize force field tactile sensing."""
        # Check required dependencies
        if self.cfg.sdf_tool in ["trimesh", "pysdf"] and not TRIMESH_AVAILABLE:
            raise ImportError(f"Trimesh is required for SDF tool '{self.cfg.sdf_tool}' but not available.")

        if self.cfg.sdf_tool == "pysdf" and not PYSDF_AVAILABLE:
            raise ImportError("PySDF is required for SDF tool 'pysdf' but not available.")

        # Find parent prims and get body handles
        self._find_body_handles()

        # Generate tactile points on elastomer surface
        self._generate_tactile_points(
            elastomer_link_name=self.cfg.elastomer_link_name,
            elastomer_tip_link_name=self.cfg.elastomer_tip_link_name,
            num_divs=[self.cfg.num_tactile_rows, self.cfg.num_tactile_cols],
            margin=getattr(self.cfg, "tactile_margin", 0.003),
            visualize=self.cfg.debug_vis and self.cfg.visualize_tactile_points,
        )

        # Initialize SDF for collision detection
        self._initialize_sdf()

        # Initialize force field data buffers
        self._initialize_force_field_buffers()

        omni.log.info(f"Force field tactile sensing initialized with {self.cfg.sdf_tool} SDF tool.")

    # TODO
    def _find_body_handles(self):
        """Find and store body handles for elastomer and indenter."""
        # Find elastomer and indenter links
        template_prim_path = self._parent_prims[0].GetPath().pathString

        # Create body views for elastomer and indenter
        elastomer_pattern = f"{template_prim_path}/{self.cfg.elastomer_link_name}"
        print(f"elastomer_pattern: {elastomer_pattern}")
        body_names_regex = [elastomer_pattern.replace("env_0", "env_*")]
        self._elastomer_body_view = self._physics_sim_view.create_rigid_body_view(body_names_regex)
        elastomer_tip_pattern = f"{template_prim_path}/{self.cfg.elastomer_tip_link_name}"
        print(f"elastomer_tip_pattern: {elastomer_tip_pattern}")
        body_names_regex = [elastomer_tip_pattern.replace("env_0", "env_*")]
        self._elastomer_tip_body_view = self._physics_sim_view.create_rigid_body_view(body_names_regex)

        # For force field sensing, we need indenter information
        if hasattr(self.cfg, "indenter_actor_name") and self.cfg.indenter_actor_name:
            # Get environment path by going up from the template_prim_path
            # template_prim_path is like "/World/envs/env_0/Robot", we want "/World/envs/env_0"
            env_path = "/".join(template_prim_path.split("/")[:-1])
            indenter_pattern = f"{env_path}/{self.cfg.indenter_actor_name}/{self.cfg.indenter_link_name}"
            print("indenter_pattern: ", indenter_pattern)
            self._indenter_body_view = self._physics_sim_view.create_rigid_body_view(
                indenter_pattern.replace(".*", "*")
            )

    def _generate_tactile_points(
        self,
        elastomer_link_name: str,
        elastomer_tip_link_name: str,
        num_divs: list = [20, 25],
        margin: float = 0.003,
        visualize: bool = False,
    ):
        """Generate tactile points on the elastomer surface.

        Args:
            elastomer_link_name: Name of the elastomer link.
            elastomer_tip_link_name: Name of the elastomer tip link.
            num_divs: Number of divisions for the tactile points [rows, cols].
            margin: Margin for the tactile points.
            visualize: Whether to visualize the points.
        """
        # First try USD-based mesh extraction
        try:
            if self._try_generate_from_usd_mesh(
                elastomer_link_name, elastomer_tip_link_name, num_divs, margin, visualize
            ):
                return
        except Exception:
            pass

        # Fallback to simple grid generation
        self._generate_simple_tactile_grid(num_divs)

    def _try_generate_from_usd_mesh(
        self, elastomer_link_name: str, elastomer_tip_link_name: str, num_divs: list, margin: float, visualize: bool
    ) -> bool:
        """Try to generate tactile points from USD mesh data."""
        from pxr import UsdGeom

        # Check if required dependencies are available
        if not TRIMESH_AVAILABLE:
            print("Trimesh not available, falling back to simple grid")
            return False

        # Get the elastomer prim path
        template_prim_path = self._parent_prims[0].GetPath().pathString
        elastomer_prim_path = f"{template_prim_path}/{elastomer_link_name}/visuals"
        print("elastomer_prim_path: ", elastomer_prim_path)

        # Find mesh prim
        mesh_prim = sim_utils.get_first_matching_child_prim(
            elastomer_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
        )

        if mesh_prim is None or not mesh_prim.IsValid():
            return False

        # Extract mesh data
        usd_mesh = UsdGeom.Mesh(mesh_prim)
        points = np.asarray(usd_mesh.GetPointsAttr().Get())
        face_indices = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get())
        print("points: ", points.shape)
        print("points: ", points[0:10, :])
        print("face_indices: ", face_indices.shape)

        # Simple triangulation
        faces = face_indices.reshape(-1, 3)

        # Create bounds
        mesh_bounds = np.array([points.min(axis=0), points.max(axis=0)])

        try:
            # Create trimesh object
            mesh = trimesh.Trimesh(vertices=points, faces=faces)

            # Generate grid on elastomer (following Isaac Gym version pattern)
            elastomer_dims = np.diff(mesh_bounds, axis=0).squeeze()
            slim_axis = np.argmin(elastomer_dims)  # Determine flat axis of elastomer

            # Get elastomer to tip transform (similar to Isaac Gym version)
            try:
                _, elastomer_to_tip_link_pos = self._get_elastomer_to_tip_transform(
                    actor_name="Robot",  # Default actor name
                    elastomer_link_name=elastomer_link_name,
                    elastomer_tip_link_name=elastomer_tip_link_name,
                )
                # elastomer_to_tip_link_pos:  tensor([[-2.5054e-11, -2.5798e-10,  2.0000e-02]], device='cuda:0')
                # print("elastomer_to_tip_link_pos: ", elastomer_to_tip_link_pos)
            except Exception as e:
                print(f"Failed to get elastomer to tip transform: {e}, using default direction")
                # Use default direction - assume tip is in positive direction of slim axis
                elastomer_to_tip_link_pos = [torch.zeros(3, device=self._device)]
                elastomer_to_tip_link_pos[0][slim_axis] = 1.0

            # Determine gap between adjacent tactile points
            axis_idxs = list(range(3))
            axis_idxs.remove(int(slim_axis))  # Remove slim idx
            div_sz = (elastomer_dims[axis_idxs] - margin * 2.0) / (np.array(num_divs) + 1)
            tactile_points_dx = min(div_sz)

            # Sample points on the flat plane (following Isaac Gym version exactly)
            planar_grid_points = []
            center = (mesh_bounds[0] + mesh_bounds[1]) / 2.0
            idx = 0
            for axis_i in range(3):
                if axis_i == slim_axis:
                    # On the slim axis, place a point far away so ray is pointing at the elastomer tip
                    planar_grid_points.append([np.sign(elastomer_to_tip_link_pos[0][slim_axis].item())])
                else:
                    axis_grid_points = np.linspace(
                        center[axis_i] - tactile_points_dx * (num_divs[idx] + 1.0) / 2.0,
                        center[axis_i] + tactile_points_dx * (num_divs[idx] + 1.0) / 2.0,
                        num_divs[idx] + 2,
                    )
                    planar_grid_points.append(axis_grid_points[1:-1])  # Leave out the extreme corners
                    idx += 1

            grid_corners = itertools.product(planar_grid_points[0], planar_grid_points[1], planar_grid_points[2])
            grid_corners = np.array(list(grid_corners))

            # Project ray in positive y direction on the mesh (following Isaac Gym version)
            mesh_data = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
            ray_dir = np.array([0, 0, 0])
            ray_dir[slim_axis] = -np.sign(elastomer_to_tip_link_pos[0][slim_axis].item())  # Ray point towards elastomer


            # Handle the ray intersection result (following Isaac Gym version exactly)
            result = mesh_data.intersects_id(
                grid_corners, np.tile([ray_dir], (grid_corners.shape[0], 1)), return_locations=True, multiple_hits=False
            )

            # Extract results based on what was returned
            if len(result) == 3:
                index_tri, index_ray, locations = result
            else:
                index_tri, index_ray = result
                locations = None

            if visualize:
                query_pointcloud = trimesh.PointCloud(locations, colors=(0.0, 0.0, 1.0))
                trimesh.Scene([mesh, query_pointcloud]).show()

            # Check if we got the expected number of points (following Isaac Gym error handling)
            if len(index_ray) != len(grid_corners):
                raise ValueError("Fewer number of tactile points than expected")

            # Sort and store tactile points (following Isaac Gym version exactly)
            tactile_points = locations[index_ray.argsort()]
            # in the frame of the elastomer
            self._tactile_pos_local = torch.tensor(tactile_points, dtype=torch.float32, device=self._device)

            # Assume tactile frame rotation are all the same (following Isaac Gym version)
            rotation = (0, 0, -np.pi)  # NOTE: assume tactile frame rotation are all the same
            rotation = R.from_euler("xyz", rotation).as_quat()
            self._tactile_quat_local = (
                torch.tensor(rotation, dtype=torch.float32, device=self._device)
                .unsqueeze(0)
                .repeat(len(tactile_points), 1)
            )

            print(f"Generated {len(tactile_points)} tactile points from USD mesh using ray casting")
            return True

        except Exception as e:
            print(f"USD-based tactile point generation failed: {e}")
            return False

    def _get_elastomer_to_tip_transform(self, actor_name: str, elastomer_link_name: str, elastomer_tip_link_name: str):
        """Get the transformation from the elastomer to the tip.

        Note: This is adapted from the Isaac Gym version to work with IsaacLab.

        Args:
            actor_name: Name of the actor.
            elastomer_link_name: Name of the elastomer link.
            elastomer_tip_link_name: Name of the elastomer tip link.

        Returns:
            tuple: Quaternion and position of the transformation.
        """
        # Get elastomer and tip body handles using IsaacLab's body view system
        if self._elastomer_body_view is None or self._elastomer_tip_body_view is None:
            raise RuntimeError("Elastomer body view not initialized")

        # Get poses directly from the dedicated body views
        elastomer_pose = self._elastomer_body_view.get_transforms()
        elastomer_tip_pose = self._elastomer_tip_body_view.get_transforms()
        elastomer_pose[..., 3:] = math_utils.convert_quat(elastomer_pose[..., 3:], to="wxyz")
        elastomer_tip_pose[..., 3:] = math_utils.convert_quat(elastomer_tip_pose[..., 3:], to="wxyz")

        # Extract poses for the first environment
        elastomer_transform = elastomer_pose[0]  # Shape: (7,) [pos + quat]
        tip_transform = elastomer_tip_pose[0]  # Shape: (7,) [pos + quat]

        # Extract positions and quaternions
        elastomer_pos = elastomer_transform[:3]  # (3,)
        elastomer_quat = elastomer_transform[3:]  # (4,)
        tip_pos = tip_transform[:3]  # (3,)
        tip_quat = tip_transform[3:]  # (4,)

        # Compute relative transform from elastomer to tip
        # Position: transform tip position to elastomer local frame
        relative_pos_world = tip_pos - elastomer_pos
        elastomer_quat_inv = math_utils.quat_inv(elastomer_quat.unsqueeze(0))
        tip_pos_local = math_utils.quat_apply(elastomer_quat_inv, relative_pos_world.unsqueeze(0))

        # Orientation: compute relative quaternion from elastomer to tip
        tip_quat_local = math_utils.quat_mul(elastomer_quat_inv, tip_quat.unsqueeze(0))

        return tip_quat_local, tip_pos_local

    def _generate_simple_tactile_grid(self, num_divs: list):
        """Generate a simple grid of tactile points."""
        print("Generating tactile points using simple grid method")

        # Create a simple grid pattern for tactile points
        x_coords = torch.linspace(-0.01, 0.01, num_divs[1], device=self._device)
        y_coords = torch.linspace(-0.01, 0.01, num_divs[0], device=self._device)

        # Create meshgrid and flatten
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing="ij")
        tactile_points = torch.stack(
            [x_grid.flatten(), torch.zeros_like(x_grid.flatten()), y_grid.flatten()], dim=1  # Z-coordinate on surface
        )

        self._tactile_pos_local = tactile_points.to(dtype=torch.float32)

        # All tactile points have the same orientation (pointing up in local frame)
        tactile_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self._device)
        self._tactile_quat_local = tactile_quat.unsqueeze(0).repeat(tactile_points.shape[0], 1)

    def _initialize_sdf(self):
        """Initialize SDF for collision detection."""
        if self.cfg.sdf_tool == "physx":
            # Create SDF Shape View of indenter for querying.
            # todo Juana: make this more general
            # prim_paths_expr = f"/World/envs/env_.*/{self.cfg.indenter_actor_name}/{self.cfg.indenter_link_name}/mesh"
            prim_paths_expr = (
                f"/World/envs/env_.*/{self.cfg.indenter_actor_name}/{self.cfg.indenter_link_name}/collisions"
            )
            # prim_paths_expr = f"/World/envs/env_.*/{self.cfg.indenter_actor_name}/{self.cfg.indenter_link_name}"
            num_query_points = self.cfg.num_tactile_rows * self.cfg.num_tactile_cols
            prepare_sdf_schemas = True
            prepare_sdf_schemas = False  # if indenter is already an SDF collision mesh

            # Create SDF Shape View of indenter for querying.
            self._indenter_sdf_view = SdfShapePrim(
                prim_paths_expr=prim_paths_expr,
                name="indenter_sdf_view",
                num_query_points=num_query_points,
                prepare_sdf_schemas=prepare_sdf_schemas,
            )
            self._indenter_sdf_view.initialize(physics_sim_view=self._physics_sim_view)

            # Compute SDF and SDF gradient
            # For PhysX SDF, we need to set up SDF shape global IDs
            self._sdf_shape_global_ids_per_env = torch.zeros(
                (self._num_envs, 1), dtype=torch.int32, device=self._device
            )
            # Implementation would set up PhysX SDF querying here
            # omni.log.warn("PhysX SDF implementation not yet complete.")

        elif self.cfg.sdf_tool in ["trimesh", "pysdf"]:
            # For mesh-based SDF tools, load indenter mesh
            if self.cfg.indenter_urdf_path is None:
                raise ValueError(f"indenter_urdf_path is required for SDF tool '{self.cfg.sdf_tool}'")

            if not URDFPY_AVAILABLE:
                raise ImportError("urdfpy is required for URDF parsing but not available.")

            # Load indenter mesh from URDF
            # This is a simplified version - full implementation would parse URDF properly
            omni.log.warn(f"Mesh-based SDF ({self.cfg.sdf_tool}) implementation not yet complete.")

    def _initialize_force_field_buffers(self):
        """Initialize data buffers for force field sensing."""
        num_tactile_points = self.cfg.num_tactile_rows * self.cfg.num_tactile_cols

        # Initialize force field data tensors
        self._data.tactile_points_pos_w = torch.zeros((self._num_envs, num_tactile_points, 3), device=self._device)
        self._data.tactile_points_quat_w = torch.zeros((self._num_envs, num_tactile_points, 4), device=self._device)
        self._data.penetration_depth = torch.zeros((self._num_envs, num_tactile_points), device=self._device)
        self._data.tactile_normal_force = torch.zeros((self._num_envs, num_tactile_points), device=self._device)
        self._data.tactile_shear_force = torch.zeros((self._num_envs, num_tactile_points, 2), device=self._device)
        self._data.contact_normals_w = torch.zeros((self._num_envs, num_tactile_points, 3), device=self._device)

    def _initialize_visualization(self):
        """Initialize visualization markers for tactile points."""
        if self.cfg.visualizer_cfg is not None:
            self._visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)

    def _setup_compliant_materials(self):
        """Setup compliant contact materials for the elastomer."""
        # Get the current stage
        stage = stage_utils.get_current_stage()

        # Create material configuration
        print("set up compliant contact materials", self.cfg.elastomer_collision_path)
        material_cfg = RigidBodyMaterialCfg(
            compliant_contact_stiffness=self.cfg.compliance_stiffness,
            compliant_contact_damping=self.cfg.compliant_damping,
        )
        self._parent_prims = sim_utils.find_matching_prims(self.cfg.prim_path.rsplit("/", 1)[0])
        self._num_envs = len(self._parent_prims)

        # Apply material to each environment
        for env_id in range(self._num_envs):
            # Construct the prim path for elastomer collision geometry
            # Get the environment path from parent prims
            if len(self._parent_prims) > env_id:
                # Use the specific environment's parent prim
                env_prim_path = self._parent_prims[env_id].GetPath().pathString

                # Construct full path to elastomer collision
                elastomer_collision_path = (
                    f"{env_prim_path}/{self.cfg.elastomer_link_name}/{self.cfg.elastomer_collision_path}"
                )

                # Spawn the rigid body material
                mat_path = spawn_rigid_body_material(elastomer_collision_path, material_cfg)

                # Get the body prim and apply the physics material
                body_prim = prim_utils.get_prim_parent(mat_path)
                physicsUtils.add_physics_material_to_prim(stage, body_prim, elastomer_collision_path)

        omni.log.warn(f"Applied compliant contact materials to {self._num_envs} environments.")

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # Convert to proper indices for internal methods
        if len(env_ids) == self._num_envs:
            internal_env_ids = slice(None)
        else:
            internal_env_ids = env_ids

        # Increment call counter for timing tracking
        self._timing_call_count += 1

        # Update camera-based tactile data
        if self.cfg.enable_camera_tactile and self._camera_sensor is not None:
            self._camera_timer.start()
            self._update_camera_tactile(internal_env_ids)
            if self._timing_call_count > 2:
                if self._timing_call_count == 3:
                    print("camera_timer: ", self._camera_timer.time_elapsed)
                self._camera_timing_total += self._camera_timer.time_elapsed
            self._camera_timer.stop()

        # Update force field tactile data
        if self.cfg.enable_force_field and self._tactile_pos_local is not None:
            self._force_field_timer.start()
            self._update_force_field(internal_env_ids)
            if self._timing_call_count > 2:
                self._force_field_timing_total += self._force_field_timer.time_elapsed
            self._force_field_timer.stop()

       

    def get_timing_summary(self) -> dict:
        """Get current timing statistics as a dictionary.

        Returns:
            Dictionary containing timing statistics.
        """
        if self._timing_call_count == 0:
            return {
                "call_count": 0,
                "camera_total": 0.0,
                "camera_average": 0.0,
                "force_field_total": 0.0,
                "force_field_average": 0.0,
                "combined_average": 0.0
            }

        ## skip the first two calls
        self._timing_call_count -= 2
        num_frames = self._timing_call_count * self._num_envs
        force_field_avg = self._force_field_timing_total / num_frames if self._force_field_timing_total > 0 else 0.0
        camera_avg = self._camera_timing_total / num_frames if self._camera_timing_total > 0 else 0.0

        return {
            "call_count": self._timing_call_count,
            "camera_total": self._camera_timing_total,
            "camera_average": camera_avg,
            "force_field_total": self._force_field_timing_total,
            "force_field_average": force_field_avg,
            "combined_average": camera_avg + force_field_avg,
            "num_envs": self._num_envs,
            "num_frames": num_frames,
            "camera_fps": 1 / camera_avg if camera_avg > 0 else 0.0,
            "force_field_fps": 1 / force_field_avg if force_field_avg > 0 else 0.0,
            "total_fps": 1 / (camera_avg + force_field_avg) if (camera_avg + force_field_avg) > 0 else 0.0,
        }

    def reset_timing_statistics(self):
        """Reset all timing statistics to zero."""
        self._camera_timing_total = 0.0
        self._force_field_timing_total = 0.0
        self._timing_call_count = 0



    def _update_camera_tactile(self, env_ids: Sequence[int] | slice):
        """Update camera-based tactile sensing data."""
        # Update camera sensor
        self._camera_sensor.update(self._sim_physics_dt)

        # Get camera data
        camera_data = self._camera_sensor.data

        if "rgb" in camera_data.output:
            self._data.tactile_rgb = camera_data.output["rgb"][env_ids].clone()

        if "distance_to_image_plane" in camera_data.output:
            self._data.tactile_depth = camera_data.output["distance_to_image_plane"][env_ids].clone()
            diff = self._nominal_tactile["distance_to_image_plane"] - self._data.tactile_depth
            self._data.taxim_tactile = self.taxim_gelsight.render_tensorized(diff.squeeze(-1))

    def _update_force_field(self, env_ids: Sequence[int] | slice):
        """Update force field tactile sensing data."""
        if self._elastomer_body_view is None or self._tactile_pos_local is None:
            return

        # Get elastomer pose
        elastomer_poses = self._elastomer_body_view.get_transforms()
        elastomer_poses[..., 3:] = math_utils.convert_quat(elastomer_poses[..., 3:], to="wxyz")

        # Transform tactile points to world coordinates
        self._transform_tactile_points_to_world(elastomer_poses[env_ids])

        # Compute penalty-based tactile forces using SDF
        if (
            self._indenter_body_view is not None
            and self._data.tactile_points_pos_w is not None
            and hasattr(self, "_indenter_sdf_view")
        ):

            # Get indenter poses
            indenter_poses = self._indenter_body_view.get_transforms()
            indenter_poses[..., 3:] = math_utils.convert_quat(indenter_poses[..., 3:], to="wxyz")

            # Get tactile points in world coordinates
            world_tactile_points = self._data.tactile_points_pos_w[env_ids]

            # Transform tactile points to indenter local frame for SDF queries
            points_indenter_local, indenter_quat_inv = self._transform_points_to_indenter_local(
                world_tactile_points, indenter_poses[env_ids]
            )

            # Query SDF for collision detection
            sdf_values_and_gradients = self._indenter_sdf_view.get_sdf_and_gradients(points_indenter_local)

            # Extract SDF values (penetration depth) and gradients (normals)
            # Ensure we have tensor data
            if isinstance(sdf_values_and_gradients, torch.Tensor):
                sdf_values = sdf_values_and_gradients[..., -1]  # Last component is SDF value
                sdf_gradients = sdf_values_and_gradients[..., :-1]  # First 3 components are gradients
            else:
                # Convert numpy array to tensor if needed
                sdf_tensor = torch.from_numpy(sdf_values_and_gradients).to(device=self._device)
                sdf_values = sdf_tensor[..., 0]
                sdf_gradients = sdf_tensor[..., 1:4]

            # Compute tactile forces
            # Todo Juana: set env_ids to all
            depth, tactile_normal_force, tactile_shear_force = self._compute_tactile_forces_from_sdf(
                points_indenter_local,
                sdf_values,
                sdf_gradients,
                world_tactile_points,
                indenter_poses[env_ids],
                indenter_quat_inv,
                elastomer_poses[env_ids],
                env_ids,
            )

            # Store results in data
            if self._data.penetration_depth is not None:
                self._data.penetration_depth[env_ids] = depth
            if self._data.tactile_normal_force is not None:
                self._data.tactile_normal_force[env_ids] = tactile_normal_force
            if self._data.tactile_shear_force is not None:
                self._data.tactile_shear_force[env_ids] = tactile_shear_force

    def _transform_tactile_points_to_world(self, elastomer_poses: torch.Tensor):
        """Transform tactile points from local to world coordinates."""
        # Extract position and quaternion from poses
        pos_w = elastomer_poses[:, :3]
        quat_w = elastomer_poses[:, 3:]

        # Transform local tactile points to world frame
        num_tactile_points = self._tactile_pos_local.shape[0]
        tactile_pos_expanded = self._tactile_pos_local.unsqueeze(0).expand(pos_w.shape[0], -1, -1)
        tactile_quat_expanded = self._tactile_quat_local.unsqueeze(0).expand(pos_w.shape[0], -1, -1)

        # Apply transformation
        tactile_pos_w = math_utils.quat_apply(
            quat_w.unsqueeze(1).expand(-1, num_tactile_points, -1), tactile_pos_expanded
        ) + pos_w.unsqueeze(1).expand(-1, num_tactile_points, -1)

        tactile_quat_w = math_utils.quat_mul(
            quat_w.unsqueeze(1).expand(-1, num_tactile_points, -1), tactile_quat_expanded
        )

        # Store in data
        self._data.tactile_points_pos_w = tactile_pos_w
        self._data.tactile_points_quat_w = tactile_quat_w

    def _transform_points_to_indenter_local(
        self, world_points: torch.Tensor, indenter_poses: torch.Tensor
    ) -> torch.Tensor:
        """Transform world coordinates to indenter local frame.

        This is used for SDF queries where you need to know contact points relative to the indenter's
        coordinate system for collision detection and force calculations.

        Args:
            world_points: Points in world coordinates. Shape: (num_envs, num_points, 3)
            indenter_poses: Indenter poses in world frame. Shape: (num_envs, 7) [pos + quat]

        Returns:
            Points in indenter local coordinates. Shape: (num_envs, num_points, 3)
        """
        # Extract position and quaternion from indenter poses
        pos_w = indenter_poses[:, :3]  # (num_envs, 3)
        quat_w = indenter_poses[:, 3:]  # (num_envs, 4)

        # Get inverse transformation (per environment)
        # wxyz in torch
        indenter_quat_inv, indenter_pos_inv = torch_utils.tf_inverse(quat_w, pos_w)

        # Compute points in the object frame
        # tf_apply will handle broadcasting: (num_envs, 4) and (num_envs, 3) -> (num_envs, num_points, 3)
        points_sdf = torch_utils.tf_apply(indenter_quat_inv, indenter_pos_inv, world_points)

        return points_sdf, indenter_quat_inv

    def _get_tactile_points_velocities(
        self, env_ids: Sequence[int] | slice, elastomer_poses: torch.Tensor
    ) -> torch.Tensor:
        """Compute the velocities of tactile points based on elastomer motion.

        This function computes the velocity of each tactile point due to the elastomer's
        linear and angular motion using the formula: v_point = v_linear + ω × r_local

        Args:
            env_ids: Environment indices to compute velocities for

        Returns:
            Tactile point velocities in world frame. Shape: (num_envs, num_points, 3)
        """
        if self._elastomer_body_view is None or self._tactile_pos_local is None:
            # Return zero velocities if not properly initialized
            num_tactile_points = self.cfg.num_tactile_rows * self.cfg.num_tactile_cols
            num_envs = len(env_ids) if isinstance(env_ids, (list, tuple)) else self._num_envs
            return torch.zeros((num_envs, num_tactile_points, 3), device=self._device)

        # Get elastomer linear and angular velocities
        elastomer_velocities = self._elastomer_body_view.get_velocities()
        elastomer_linvel_world = elastomer_velocities[env_ids, :3]  # Linear velocity
        elastomer_angvel_world = elastomer_velocities[env_ids, 3:]  # Angular velocity

        # Get elastomer orientation
        elastomer_quat_world = elastomer_poses[env_ids, 3:]  # Quaternion

        num_tactile_points = self._tactile_pos_local.shape[0]

        # Get number of environments
        if isinstance(env_ids, slice):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)

        # Expand quaternions for transformation
        quat_expanded = elastomer_quat_world.unsqueeze(1).expand(num_envs, num_tactile_points, 4)

        # Transform local positions to world frame relative vectors
        tactile_pos_world_relative = math_utils.quat_apply(
            quat_expanded, self._tactile_pos_local.expand(num_envs, num_tactile_points, 3)
        )

        # Compute velocity due to angular motion: ω × r
        angvel_expanded = elastomer_angvel_world.unsqueeze(1).expand(num_envs, num_tactile_points, 3)
        angular_velocity_contribution = torch.cross(angvel_expanded, tactile_pos_world_relative, dim=-1)

        # Add linear velocity contribution
        linvel_expanded = elastomer_linvel_world.unsqueeze(1).expand(num_envs, num_tactile_points, 3)
        tactile_velocity_world = angular_velocity_contribution + linvel_expanded

        return tactile_velocity_world

    def _compute_tactile_forces_from_sdf(
        self,
        points_indenter_local: torch.Tensor,
        sdf_values: torch.Tensor,
        sdf_gradients: torch.Tensor,
        world_tactile_points: torch.Tensor,
        indenter_poses: torch.Tensor,
        indenter_quat_inv: torch.Tensor,
        elastomer_poses: torch.Tensor,
        env_ids: Sequence[int] | slice,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute tactile forces from SDF values and gradients.

        Based on the penalty-based tactile force computation from Isaac Gym version.

        Args:
            sdf_values: SDF values (negative means penetration). Shape: (num_envs, num_points)
            sdf_gradients: SDF gradients (surface normals). Shape: (num_envs, num_points, 3)
            world_tactile_points: Tactile points in world coordinates. Shape: (num_envs, num_points, 3)
            indenter_poses: Indenter poses in world frame. Shape: (num_envs, 7)
            env_ids: Environment indices being updated

        Returns:
            tuple: (penetration_depth, tactile_normal_force, tactile_shear_force)
        """
        # Convert SDF values to penetration depth (positive for penetration)
        depth = torch.clamp(-sdf_values, min=0.0)  # Negative SDF means inside (penetrating)

        # Convert flat index back to (env_idx, point_idx)
        num_points = depth.shape[1]

        # Get collision mask for points that are penetrating
        collision_mask = depth > 0.0

        # Initialize force tensors
        num_points = world_tactile_points.shape[1]
        num_envs = world_tactile_points.shape[0]

        tactile_normal_force = torch.zeros((num_envs, num_points), device=self._device)
        tactile_shear_force = torch.zeros((num_envs, num_points, 2), device=self._device)

        if collision_mask.any():
            # Normalize gradients to get surface normals in local frame
            normals_local = torch.nn.functional.normalize(sdf_gradients, dim=-1)

            # Transform normals to world frame (rotate by indenter orientation)
            indenter_quat = indenter_poses[:, 3:]  # (num_envs, 4) wxyz

            # Apply quaternion transformation
            normals_world = math_utils.quat_apply(indenter_quat, normals_local)

            # Compute normal contact force: F_n = k_n * depth
            fc_norm = self.cfg.tactile_kn * depth
            fc_world = fc_norm.unsqueeze(-1) * normals_world

            # Get tactile point velocities based on elastomer motion
            tactile_velocity_world = self._get_tactile_points_velocities(env_ids, elastomer_poses)

            sdf_vel_world = self._indenter_body_view.get_velocities()
            sdf_lin_world = sdf_vel_world[env_ids, :3]
            sdf_ang_world = sdf_vel_world[env_ids, 3:]

            closest_points_sdf = points_indenter_local + depth.unsqueeze(-1) * normals_local
            sdf_lin_world = sdf_lin_world.unsqueeze(1).expand(num_envs, num_points, 3)
            sdf_ang_world = sdf_ang_world.unsqueeze(1).expand(num_envs, num_points, 3)
            closest_points_vel_world = (
                torch.cross(sdf_ang_world, math_utils.quat_apply(indenter_poses[env_ids, 3:], closest_points_sdf))
                + sdf_lin_world
            )

            # Compute relative velocity at contact points
            relative_velocity_world = tactile_velocity_world - closest_points_vel_world

            # Compute tangential velocity (perpendicular to normal)
            vt_world = relative_velocity_world - normals_world * torch.sum(
                normals_world * relative_velocity_world, dim=-1, keepdim=True
            )  # todo
            vt_norm = torch.norm(vt_world, dim=-1)

            # Compute friction force: F_t = min(k_t * |v_t|, mu * F_n)
            ft_static_norm = self.cfg.tactile_kt * vt_norm
            ft_dynamic_norm = self.cfg.tactile_mu * fc_norm
            ft_norm = torch.minimum(ft_static_norm, ft_dynamic_norm)

            # Apply friction force opposite to tangential velocity
            ft_world = -ft_norm.unsqueeze(-1) * vt_world / (vt_norm.unsqueeze(-1).clamp(min=1e-9))

            # Total tactile force in world frame
            tactile_force_world = fc_world + ft_world

            # Transform forces to tactile frame (using tactile point orientations)
            if self._data.tactile_points_quat_w is not None:
                tactile_quat_inv = math_utils.quat_inv(self._data.tactile_points_quat_w[env_ids])
                #  '''tactile force in tactile frame'''
                tactile_force_tactile = math_utils.quat_apply(tactile_quat_inv, tactile_force_world)

                # Extract normal and shear components
                # Assume tactile frame has Z as normal direction
                tactile_normal_force = tactile_force_tactile[..., 2]  # Z component
                tactile_shear_force = tactile_force_tactile[..., :2]  # X,Y components

        return depth, tactile_normal_force, tactile_shear_force

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects."""
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if self._tactile_visualizer is None:
                self._tactile_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self._tactile_visualizer.set_visibility(True)
        else:
            if self._tactile_visualizer is not None:
                self._tactile_visualizer.set_visibility(False)

    #########################################################################################
    # Debug visualization
    #########################################################################################
    def _debug_vis_callback(self, event):
        """Callback for debug visualization.

        Visualizes tactile points with different colors/markers based on:
        - Contact state (in contact vs. not in contact)
        - Force magnitude (color intensity based on force)
        """
        # Safety check - return if not properly initialized
        if (
            self._data.tactile_points_pos_w is None
            or not hasattr(self, "_tactile_visualizer")
            or self._tactile_visualizer is None
        ):
            return

        # print("tactile_points_pos_w", self._data.tactile_points_pos_w)
        # Get tactile points positions in world frame
        tactile_points_w = self._data.tactile_points_pos_w  # Shape: (num_envs, num_points, 3)

        if tactile_points_w.numel() == 0:
            return

        # Flatten points for visualization (all environments)
        viz_points = tactile_points_w.view(-1, 3)  # Shape: (num_envs * num_points, 3)

        # Initialize marker indices (0: no contact, 1: weak contact, 2: strong contact)
        marker_indices = torch.zeros(viz_points.shape[0], dtype=torch.long, device=self._device)

        # # Visualize tactile points with basic scaling
        marker_scales = torch.ones(viz_points.shape[0], 3, device=self._device)
        # if self._data.tactile_normal_force is not None and normal_forces.numel() > 0:
        #     # Scale markers based on force (clamp to reasonable range)
        #     force_tensor = normal_forces * 10.0
        #     force_scale = torch.maximum(torch.minimum(force_tensor, torch.tensor(3.0, device=self._device)),
        #                                 torch.tensor(0.5, device=self._device))
        #     marker_scales = marker_scales * force_scale.unsqueeze(-1)

        # Visualize tactile points
        self._tactile_visualizer.visualize(translations=viz_points, marker_indices=marker_indices, scales=marker_scales)
