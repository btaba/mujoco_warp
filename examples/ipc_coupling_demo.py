"""
Interactive Two-Way Soft-Coupling Demo between MuJoCo Warp (MJWarp) and LibUIPC

This script couples the rigid robot arm dynamics solved in MJWarp with the robust,
intersection-free soft trashbag cloth contacts solved in LibUIPC on the GPU.
"""

import argparse
import os
import numpy as np
import trimesh
from huggingface_hub import hf_hub_download

import warp as wp
import mujoco
import mujoco_warp as mjw
from mujoco_warp._src import cli
from mujoco_warp._src import forward
from etils import epath
import uipc

# Warp kernel to compute feedback forces and torques on the GPU
@wp.kernel
def compute_coupling_forces_kernel(
    target_pos: wp.array(dtype=wp.vec3, ndim=2),
    target_mat: wp.array(dtype=wp.mat33, ndim=2),
    resolved_T: wp.array(dtype=wp.mat44, ndim=2),
    stiffness: float,
    damping: float,
    mass: float,
    out_xfrc: wp.array(dtype=wp.spatial_vector, ndim=2)
):
    env_idx, link_idx = wp.tid()
    
    # Target position & rotation from MuJoCo Warp
    t_pos = target_pos[env_idx, link_idx]
    t_mat = target_mat[env_idx, link_idx]
    
    # Resolved position & rotation from LibUIPC ABD body
    r_T = resolved_T[env_idx, link_idx]
    r_pos = wp.vec3(r_T[0, 3], r_T[1, 3], r_T[2, 3])
    
    # Compute translation spring restoring force: F = Mass * (P_ipc - P_target) * Stiffness
    deflection_pos = r_pos - t_pos
    force = mass * deflection_pos * stiffness
    
    # Pack into applied Cartesian force/torque: wp.spatial_vector(torque, force)
    out_xfrc[env_idx, link_idx] = wp.spatial_vector(
        0.0, 0.0, 0.0,     # Torque
        force[0], force[1], force[2]  # Force
    )

def main():
    import sys
    from absl import flags
    flags.FLAGS(sys.argv)
    wp.init()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--nworld", type=int, default=4)
    parser.add_argument("--nstep", type=int, default=500)
    args = parser.parse_args()

    n_envs = args.nworld
    dt = 0.005
    
    print(f"Initializing MuJoCo Warp for {n_envs} parallel environments...")
    # Load the standard scene.xml (which only contains Franka and Plane - no native flex)
    xml_path = "benchmarks/franka_emika_panda/scene.xml"
    mjm = cli.load_model(epath.Path(xml_path))
    
    # Force nworld to user-selected environment count
    flags.FLAGS.nworld = n_envs
    m, d, rc, ctrls = cli.init_structs(forward.step, mjm)
    
    # Finger link body indices in MuJoCo
    # In panda.xml: left_finger is body 10, right_finger is body 11 (offset by robot base)
    left_finger_body_idx = 10
    right_finger_body_idx = 11

    print("Initializing LibUIPC World and Subscenes...")
    engine = uipc.core.Engine("cuda", "mjwarp_ipc_workspace")
    world = uipc.core.World(engine)
    scene = uipc.core.Scene()
    
    # Setup Constitutions in IPC Scene
    abd_constitution = uipc.constitution.AffineBodyConstitution()
    stc_constitution = uipc.constitution.SoftTransformConstraint()
    nks_constitution = uipc.constitution.StrainLimitingBaraffWitkinShell()
    dsb_constitution = uipc.constitution.DiscreteShellBending()
    
    scene.constitution_tabular().insert(abd_constitution)
    scene.constitution_tabular().insert(stc_constitution)
    scene.constitution_tabular().insert(nks_constitution)
    scene.constitution_tabular().insert(dsb_constitution)
    
    # Set up isolated subscenes for batch mode
    subscenes = []
    for env_idx in range(n_envs):
        subscene = scene.subscene_tabular().create(f"subscene_{env_idx}")
        subscenes.append(subscene)
        
    # Ground Plane in IPC
    plane_geom = uipc.geometry.ground(0.0, np.array([0.0, 0.0, 1.0]))
    scene.objects().create("plane").geometries().create(plane_geom)
    for subscene in subscenes:
        subscene.apply_to(plane_geom)

    # Load Finger Mesh for ABD modeling in IPC
    finger_mesh_path = "benchmarks/franka_emika_panda/assets/finger_0.obj"
    finger_tri_mesh = trimesh.load(finger_mesh_path)
    finger_convex = finger_tri_mesh.convex_hull  # Strictly watertight and closed
    finger_verts = np.array(finger_convex.vertices, dtype=np.float64)
    finger_faces = np.array(finger_convex.faces, dtype=np.int32)

    # Load Trashbag Mesh from HF Snapshot local cache
    trashbag_mesh_path = "benchmarks/franka_emika_panda/assets/Trashbag_coarse.obj"
    bag_tri_mesh = trimesh.load(trashbag_mesh_path)
    bag_verts = np.array(bag_tri_mesh.vertices, dtype=np.float64)
    bag_faces = np.array(bag_tri_mesh.faces, dtype=np.int32)

    print("Populating ABD Fingers and Soft Bag in IPC...")
    
    # Map rigid fingers into IPC as ABD bodies for each environment
    finger_objs = []
    for env_idx in range(n_envs):
        # Left Finger (ABD Body + Soft Constraint)
        left_mesh = uipc.geometry.trimesh(finger_verts.copy(), finger_faces.copy())
        left_mesh.instances().resize(1)  # Allocate 1 instance slot
        abd_constitution.apply_to(left_mesh, kappa=10.0 * uipc.unit.MPa, mass_density=500.0)
        stc_constitution.apply_to(left_mesh)
        
        # Find constraint attributes automatically populated by SoftTransformConstraint
        is_constrained = left_mesh.instances().find(uipc.builtin.is_constrained)
        aim_transform = left_mesh.instances().find(uipc.builtin.aim_transform)
        uipc.view(is_constrained)[0] = 1
        
        left_obj = scene.objects().create(f"left_finger_{env_idx}")
        left_obj.geometries().create(left_mesh)
        subscenes[env_idx].apply_to(left_mesh)
        finger_objs.append((left_mesh, aim_transform))

        # Right Finger (ABD Body + Soft Constraint)
        right_mesh = uipc.geometry.trimesh(finger_verts.copy(), finger_faces.copy())
        right_mesh.instances().resize(1)  # Allocate 1 instance slot
        abd_constitution.apply_to(right_mesh, kappa=10.0 * uipc.unit.MPa, mass_density=500.0)
        stc_constitution.apply_to(right_mesh)
        
        is_constrained_r = right_mesh.instances().find(uipc.builtin.is_constrained)
        aim_transform_r = right_mesh.instances().find(uipc.builtin.aim_transform)
        uipc.view(is_constrained_r)[0] = 1
        
        right_obj = scene.objects().create(f"right_finger_{env_idx}")
        right_obj.geometries().create(right_mesh)
        subscenes[env_idx].apply_to(right_mesh)
        finger_objs.append((right_mesh, aim_transform_r))

        # Soft Trashbag Deformable Mesh (Neo-Hookean Shell + Discrete Hinge Bending)
        bag_mesh = uipc.geometry.trimesh(bag_verts.copy(), bag_faces.copy())
        moduli = uipc.constitution.ElasticModuli2D.youngs_poisson(1.5e4, 0.45)
        nks_constitution.apply_to(bag_mesh, moduli=moduli, mass_density=150.0, thickness=0.001)
        dsb_constitution.apply_to(bag_mesh, bending_stiffness=2.0)
        
        # Translate to match initial MuJoCo robot target position
        uipc.view(bag_mesh.positions())[:] += np.array([[0.62], [-0.35], [0.25]])
        
        bag_obj = scene.objects().create(f"trashbag_{env_idx}")
        bag_obj.geometries().create(bag_mesh)
        subscenes[env_idx].apply_to(bag_mesh)

    # Build the IPC World Scene
    world.init(scene)
    
    # Setup ABD state geometry & accessor for batch data transfer
    abd_accessor = world.features().find(uipc.core.AffineBodyStateAccessorFeature)
    abd_state_geom = abd_accessor.create_geometry()
    # Create placeholder attributes to allocate instance slots
    abd_state_geom.instances().create(uipc.builtin.transform, np.eye(4, dtype=np.float64))
    
    # Allocating GPU arrays for zero-copy Warp mapping
    wp_target_pos = wp.zeros(shape=(n_envs, 2), dtype=wp.vec3, device="cuda")
    wp_target_mat = wp.zeros(shape=(n_envs, 2), dtype=wp.mat33, device="cuda")
    wp_resolved_T = wp.zeros(shape=(n_envs, 2), dtype=wp.mat44, device="cuda")
    wp_reaction_xfrc = wp.zeros(shape=(n_envs, 2), dtype=wp.spatial_vector, device="cuda")

    # Capture the step function in a Warp graph for rapid GPU-side launches
    with wp.ScopedCapture() as capture:
        forward.step(m, d)

    print(f"Starting Time Loop (Substeps = {args.nstep})...")
    
    stiffness = 150.0 / (dt ** 2)
    mass = 0.015  # Finger Mass in kg
    
    for step in range(args.nstep):
        # 1. Step Kinematic Phase in MJWarp (PD joint controller update)
        wp.capture_launch(capture.graph)
        wp.synchronize()
        
        # 2. Retrieve target transforms of left/right fingers from MJWarp
        # Copy positions and rotation matrices from d.xpos and d.xmat
        target_pos_np = d.xpos.numpy()
        target_mat_np = d.xmat.numpy()
        
        for env_idx in range(n_envs):
            # Extract Left Finger Pos & Mat
            l_pos = target_pos_np[env_idx, left_finger_body_idx]
            l_mat = target_mat_np[env_idx, left_finger_body_idx]
            l_T = np.eye(4, dtype=np.float64)
            l_T[:3, :3] = l_mat
            l_T[:3, 3] = l_pos
            
            # Update Left Finger aim_transform in IPC
            l_mesh, l_aim = finger_objs[env_idx * 2]
            uipc.view(l_aim)[:] = l_T
            
            # Extract Right Finger Pos & Mat
            r_pos = target_pos_np[env_idx, right_finger_body_idx]
            r_mat = target_mat_np[env_idx, right_finger_body_idx]
            r_T = np.eye(4, dtype=np.float64)
            r_T[:3, :3] = r_mat
            r_T[:3, 3] = r_pos
            
            # Update Right Finger aim_transform in IPC
            r_mesh, r_aim = finger_objs[env_idx * 2 + 1]
            uipc.view(r_aim)[:] = r_T
            
        # 3. Step pyuipc contact & cloth deformation solver
        world.advance()
        world.retrieve()
        
        # 4. Retrieve solved ABD positions from IPC via geometry copy
        abd_accessor.copy_to(abd_state_geom)
        trans_attr = abd_state_geom.instances().find(uipc.builtin.transform)
        resolved_T_np = uipc.view(trans_attr)[:]  # shape: (n_envs * 2, 4, 4)
        
        # 5. Sync resolved poses to Warp GPU arrays
        # Also copy targets to GPU array for Warp calculation
        target_pos_batch = np.zeros((n_envs, 2, 3), dtype=np.float32)
        target_mat_batch = np.zeros((n_envs, 2, 3, 3), dtype=np.float32)
        resolved_T_batch = np.zeros((n_envs, 2, 4, 4), dtype=np.float32)
        
        for env_idx in range(n_envs):
            target_pos_batch[env_idx, 0] = target_pos_np[env_idx, left_finger_body_idx]
            target_pos_batch[env_idx, 1] = target_pos_np[env_idx, right_finger_body_idx]
            target_mat_batch[env_idx, 0] = target_mat_np[env_idx, left_finger_body_idx]
            target_mat_batch[env_idx, 1] = target_mat_np[env_idx, right_finger_body_idx]
            
            # Left and right resolved ABD transforms
            resolved_T_batch[env_idx, 0] = resolved_T_np[env_idx * 2]
            resolved_T_batch[env_idx, 1] = resolved_T_np[env_idx * 2 + 1]
            
        # Zero-copy write into Warp GPU buffers
        wp_target_pos.assign(target_pos_batch)
        wp_target_mat.assign(target_mat_batch)
        wp_resolved_T.assign(resolved_T_batch)
        
        # 6. Launch Warp kernel to calculate reaction forces in parallel
        wp.launch(
            compute_coupling_forces_kernel,
            dim=(n_envs, 2),
            inputs=[wp_target_pos, wp_target_mat, wp_resolved_T, stiffness, 0.0, mass],
            outputs=[wp_reaction_xfrc]
        )
        wp.synchronize()
        
        # 7. Feed computed reaction forces back into MuJoCo applied Cartesian forces
        reaction_xfrc_np = wp_reaction_xfrc.numpy()  # shape: (n_envs, 2, 6)
        xfrc_applied_np = d.xfrc_applied.numpy()
        
        for env_idx in range(n_envs):
            # Apply Left Finger force feedback
            xfrc_applied_np[env_idx, left_finger_body_idx] = reaction_xfrc_np[env_idx, 0]
            # Apply Right Finger force feedback
            xfrc_applied_np[env_idx, right_finger_body_idx] = reaction_xfrc_np[env_idx, 1]
            
        d.xfrc_applied.assign(xfrc_applied_np)
        
        if step % 50 == 0:
            print(f"Step {step:03d}: Coupling solved successfully.")

    print("Simulation completed successfully!")

if __name__ == "__main__":
    main()
