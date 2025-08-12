#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to demonstrate how to inspect joint types (prismatic vs revolute) in IsaacLab.

This script shows multiple methods to determine if a joint is prismatic:
1. Through USD prim inspection
2. Through joint names and patterns
3. Through articulation data
4. Through URDF/USD file inspection

Usage:
    ./isaaclab.sh -p scripts/inspect_joint_types.py
"""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments  
parser = argparse.ArgumentParser(description="Inspect joint types in IsaacLab articulations.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import isaacsim.core.utils.prims as prim_utils
from pxr import UsdPhysics, Usd

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab_assets import CARTPOLE_CFG, FRANKA_PANDA_CFG

def inspect_joint_types_via_usd(articulation: Articulation):
    """Method 1: Inspect joint types directly from USD prims."""
    print("\n" + "="*60)
    print("METHOD 1: USD Prim Inspection")
    print("="*60)
    
    # Get the USD stage
    stage = prim_utils.get_current_stage()
    
    # Get the root prim of the articulation
    root_prim_path = articulation.cfg.prim_path.split("/env_0")[0] + "/env_0"  # Handle regex paths
    root_prim = stage.GetPrimAtPath(root_prim_path)
    
    print(f"Inspecting articulation at: {root_prim_path}")
    
    # Recursively find all joint prims
    def find_joints(prim):
        joints = []
        # Check if current prim is a joint
        if prim.IsA(UsdPhysics.Joint):
            joints.append(prim)
        
        # Recursively check children
        for child in prim.GetAllChildren():
            joints.extend(find_joints(child))
        
        return joints
    
    joint_prims = find_joints(root_prim)
    
    for joint_prim in joint_prims:
        joint_name = joint_prim.GetName()
        joint_path = joint_prim.GetPath()
        
        # Check joint type using USD APIs
        if joint_prim.IsA(UsdPhysics.PrismaticJoint):
            joint_type = "PRISMATIC"
            # Get axis direction for prismatic joints
            if joint_prim.HasAttribute("physics:axis"):
                axis = joint_prim.GetAttribute("physics:axis").Get()
                print(f"  📐 {joint_name:20} -> {joint_type:10} (axis: {axis})")
            else:
                print(f"  📐 {joint_name:20} -> {joint_type:10}")
                
        elif joint_prim.IsA(UsdPhysics.RevoluteJoint):
            joint_type = "REVOLUTE"
            # Get axis direction for revolute joints  
            if joint_prim.HasAttribute("physics:axis"):
                axis = joint_prim.GetAttribute("physics:axis").Get()
                print(f"  🔄 {joint_name:20} -> {joint_type:10} (axis: {axis})")
            else:
                print(f"  🔄 {joint_name:20} -> {joint_type:10}")
                
        elif joint_prim.IsA(UsdPhysics.FixedJoint):
            joint_type = "FIXED"
            print(f"  🔒 {joint_name:20} -> {joint_type:10}")
        else:
            joint_type = "UNKNOWN"
            print(f"  ❓ {joint_name:20} -> {joint_type:10}")

def inspect_joint_types_via_articulation_data(articulation: Articulation):
    """Method 2: Inspect joint types through articulation data and joint names."""
    print("\n" + "="*60)
    print("METHOD 2: Articulation Data Analysis")
    print("="*60)
    
    # Get joint names
    joint_names = articulation.joint_names
    
    print(f"Total joints: {len(joint_names)}")
    print(f"Joint names: {joint_names}")
    
    # Common patterns for identifying prismatic joints by name
    prismatic_patterns = [
        "slider", "prismatic", "linear", "finger", "gripper", 
        "dummy_base_prismatic", "cart", "slide"
    ]
    
    revolute_patterns = [
        "joint", "revolute", "hinge", "rotation", "spin",
        "dummy_base_revolute", "pole", "arm", "shoulder"
    ]
    
    print("\nJoint type inference from names:")
    for i, joint_name in enumerate(joint_names):
        joint_name_lower = joint_name.lower()
        
        # Check for prismatic patterns
        is_prismatic = any(pattern in joint_name_lower for pattern in prismatic_patterns)
        is_revolute = any(pattern in joint_name_lower for pattern in revolute_patterns)
        
        if is_prismatic and not is_revolute:
            print(f"  📐 {joint_name:20} -> LIKELY PRISMATIC")
        elif is_revolute and not is_prismatic:
            print(f"  🔄 {joint_name:20} -> LIKELY REVOLUTE") 
        else:
            print(f"  ❓ {joint_name:20} -> UNCERTAIN (check USD)")

def inspect_joint_limits_and_properties(articulation: Articulation):
    """Method 3: Inspect joint properties that can hint at joint type."""
    print("\n" + "="*60)
    print("METHOD 3: Joint Properties Analysis")
    print("="*60)
    
    # Get joint position limits
    joint_pos_limits = articulation.data.soft_joint_pos_limits
    joint_names = articulation.joint_names
    
    print("Joint limits analysis (can help identify prismatic vs revolute):")
    for i, joint_name in enumerate(joint_names):
        lower_limit = joint_pos_limits[0, i].item()
        upper_limit = joint_pos_limits[0, i].item() if joint_pos_limits.shape[0] > 0 else 0.0
        range_size = upper_limit - lower_limit
        
        # Heuristics based on typical joint limits:
        # - Prismatic joints often have smaller ranges (e.g., 0-0.04 for gripper)
        # - Revolute joints often have larger ranges (e.g., -pi to pi)
        if range_size < 1.0:  # Less than 1 meter/radian
            hint = "POSSIBLY PRISMATIC (small range)"
        elif range_size > 6.0:  # More than ~2*pi radians
            hint = "POSSIBLY REVOLUTE (large range)"
        else:
            hint = "UNCERTAIN"
            
        print(f"  {joint_name:20} -> Range: [{lower_limit:6.3f}, {upper_limit:6.3f}] ({range_size:6.3f}) -> {hint}")

def inspect_via_drive_properties(articulation: Articulation):
    """Method 4: Inspect joint drive properties to determine type."""
    print("\n" + "="*60)
    print("METHOD 4: Drive Properties Analysis")
    print("="*60)
    
    # Get the USD stage and find joint prims
    stage = prim_utils.get_current_stage()
    root_prim_path = articulation.cfg.prim_path.split("/env_0")[0] + "/env_0"
    root_prim = stage.GetPrimAtPath(root_prim_path)
    
    def find_joints_recursive(prim):
        joints = []
        if prim.IsA(UsdPhysics.Joint):
            joints.append(prim)
        for child in prim.GetAllChildren():
            joints.extend(find_joints_recursive(child))
        return joints
    
    joint_prims = find_joints_recursive(root_prim)
    
    print("Drive API analysis:")
    for joint_prim in joint_prims:
        joint_name = joint_prim.GetName()
        
        # Check for linear drive (prismatic) vs angular drive (revolute)
        has_linear_drive = joint_prim.HasAPI(UsdPhysics.DriveAPI, "linear")
        has_angular_drive = joint_prim.HasAPI(UsdPhysics.DriveAPI, "angular")
        
        if has_linear_drive:
            print(f"  📐 {joint_name:20} -> HAS LINEAR DRIVE (prismatic)")
        elif has_angular_drive:
            print(f"  🔄 {joint_name:20} -> HAS ANGULAR DRIVE (revolute)")
        else:
            print(f"  ❓ {joint_name:20} -> NO DRIVE API")

def main():
    """Main function."""
    # Initialize simulation
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    # Set camera view
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    
    # Design scene with ground plane and lighting
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    
    # Create origins for different robots
    prim_utils.create_prim("/World/Origin1", "Xform", translation=[0.0, 0.0, 0.0])
    prim_utils.create_prim("/World/Origin2", "Xform", translation=[2.0, 0.0, 0.0])
    
    # Spawn Cartpole (has prismatic joint for cart movement)
    print("\n🤖 ANALYZING CARTPOLE ROBOT")
    print("="*80)
    cartpole_cfg = CARTPOLE_CFG.replace(prim_path="/World/Origin1/Robot")
    cartpole = Articulation(cfg=cartpole_cfg)
    
    # Spawn Franka Panda (has prismatic joints for gripper)
    print("\n🤖 ANALYZING FRANKA PANDA ROBOT")  
    print("="*80)
    franka_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Origin2/Robot")
    franka = Articulation(cfg=franka_cfg)
    
    # Play simulation to initialize everything
    sim.reset()
    
    print("\n🔍 CARTPOLE ANALYSIS:")
    inspect_joint_types_via_usd(cartpole)
    inspect_joint_types_via_articulation_data(cartpole)
    inspect_joint_limits_and_properties(cartpole)
    inspect_via_drive_properties(cartpole)
    
    print("\n🔍 FRANKA PANDA ANALYSIS:")
    inspect_joint_types_via_usd(franka)
    inspect_joint_types_via_articulation_data(franka)
    inspect_joint_limits_and_properties(franka)
    inspect_via_drive_properties(franka)
    
    print("\n" + "="*80)
    print("🎯 SUMMARY: How to identify prismatic joints")
    print("="*80)
    print("1. USD Prim Check: prim.IsA(UsdPhysics.PrismaticJoint)")
    print("2. Joint Names: Look for patterns like 'slider', 'prismatic', 'finger', 'cart'")
    print("3. Joint Limits: Prismatic joints often have smaller ranges (0-0.04m for grippers)")
    print("4. Drive API: Check for 'linear' drive vs 'angular' drive")
    print("5. Configuration: Check actuator configs for prismatic joint names")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
        # Keep simulation running briefly to see the robots
        import time
        time.sleep(3)
    finally:
        simulation_app.close()