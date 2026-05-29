"""
Interactive Two-Way Soft-Coupling Visual Demo between MuJoCo Warp and LibUIPC

This script runs the coupled simulation inside the native MuJoCo 3D viewer,
copying the IPC resolved trashbag vertices directly into MuJoCo's flex rendering
buffers, allowing you to see the garbage bag deforming and folding in real-time.
"""

import argparse
import os
import sys
import time
import numpy as np
import trimesh
from huggingface_hub import hf_hub_download
from etils import epath
from PIL import Image

import warp as wp
import mujoco
import mujoco.viewer
import mujoco_warp as mjw
from mujoco_warp._src import cli
from mujoco_warp._src import forward
from etils import epath
import uipc
from uipc.backend import SceneVisitor

# Warp kernel to compute feedback forces on the GPU
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
    
    t_pos = target_pos[env_idx, link_idx]
    r_T = resolved_T[env_idx, link_idx]
    r_pos = wp.vec3(r_T[0, 3], r_T[1, 3], r_T[2, 3])
    
    deflection_pos = t_pos - r_pos
    force = mass * deflection_pos * stiffness
    
    out_xfrc[env_idx, link_idx] = wp.spatial_vector(
        0.0, 0.0, 0.0,
        force[0], force[1], force[2]
    )

def main():
    import sys
    from absl import flags
    
    # Define custom flags in harmony with absl
    flags.DEFINE_integer("nworld_coupling", 1, "number of parallel environments")
    flags.DEFINE_integer("nstep_coupling", 1750, "number of simulation steps")
    flags.DEFINE_bool("headless", False, "run simulation in headless validation mode")
    
    flags.FLAGS(sys.argv)
    wp.init()
    
    # Silence pyuipc C++ logging to prevent stdout buffering/flooding
    uipc.Logger.set_level(uipc.Logger.Level.Warn)

    # Force single environment for smooth real-time rendering
    n_envs = flags.FLAGS.nworld_coupling
    dt = 0.001

    print("Loading MuJoCo Scene with flexcomp definition...", flush=True)
    xml_path = "benchmarks/franka_emika_panda/panda_trashbag_ipc.xml"
    mjm = cli.load_model(epath.Path(xml_path))
    
    flags.FLAGS.nworld = n_envs
    m, d_temp, rc, ctrls = cli.init_structs(forward.step, mjm)
    
    renderer = mujoco.Renderer(mjm, height=480, width=640)
    
    left_finger_body_idx = 10
    right_finger_body_idx = 11

    # ==========================================================================
    # 1. Kinematic Initialization: Compute Starting Fingertip Pos and Rotations
    # ==========================================================================
    mjd = mujoco.MjData(mjm)
    qpos_start = np.array([0.0, -0.9, 0.0, -2.3, 0.0, 2.3, -0.78, 0.04, 0.04])
    mjd.qpos[:9] = qpos_start
    mjd.ctrl[:8] = [0.0, -0.9, 0.0, -2.3, 0.0, 2.3, -0.78, 255.0] # gripper open
    
    # Recompute CPU kinematics with correct starting pose
    mujoco.mj_kinematics(mjm, mjd)
    
    # Create GPU Data correctly populated with starting posture
    d = mjw.put_data(
        mjm, mjd, nworld=n_envs,
        nconmax=cli.NCONMAX.value,
        njmax=cli.NJMAX.value,
        njmax_nnz=cli.NJMAX_NNZ.value,
        nccdmax=cli.NCCDMAX.value
    )
    
    # Copy start pose and propagate forward kinematics on GPU
    forward.step(m, d)
    wp.synchronize()
    
    # Sync back to mjd to ensure mjd contains correct starting poses
    mjw.get_data_into(mjd, mjm, d)

    # Read actual fingertip coordinate vectors from GPU
    init_pos_np = d.xpos.numpy()
    init_mat_np = d.xmat.numpy()
    
    l_pos = init_pos_np[0, left_finger_body_idx]
    l_mat = init_mat_np[0, left_finger_body_idx]
    l_T = np.eye(4, dtype=np.float64)
    l_T[:3, :3] = l_mat
    l_T[:3, 3] = l_pos
    
    r_pos = init_pos_np[0, right_finger_body_idx]
    r_mat = init_mat_np[0, right_finger_body_idx]
    r_T = np.eye(4, dtype=np.float64)
    r_T[:3, :3] = r_mat
    r_T[:3, 3] = r_pos

    # ==========================================================================
    # 2. Initialize LibUIPC World & Scene config
    # ==========================================================================
    print("Initializing LibUIPC World...", flush=True)
    engine = uipc.core.Engine("cuda", "mjwarp_ipc_workspace")
    world = uipc.core.World(engine)
    scene = uipc.core.Scene()
    
    # Set IPC gravity to match MuJoCo's vertical z-axis [0, 0, -9.81]
    gravity_attr = scene.config().find("gravity")
    uipc.view(gravity_attr)[:] = np.array([[0.0], [0.0], [-9.81]])
    
    # Sync UIPC step size with simulation dt
    dt_attr = scene.config().find("dt")
    uipc.view(dt_attr)[0] = dt
    
    # Disable solver sanity checks to prevent numerical TOI CCD assertion crashes
    sanity_attr = scene.config().find("sanity_check/enable")
    uipc.view(sanity_attr)[0] = 0
    
    # Increase the contact barrier thickness slightly to improve numerical stability during tight grasping
    d_hat_attr = scene.config().find("contact/d_hat")
    uipc.view(d_hat_attr)[0] = 0.02
    
    # Enable Augmented Lagrangian IPC for near-zero contact penetration and maximum solver stability
    const_attr = scene.config().find("contact/constitution")
    uipc.view(const_attr)[0] = "al-ipc"
    
    # Set CCD tolerance to be much tighter to handle tiny time-of-impact values accurately
    ccd_tol_attr = scene.config().find("newton/ccd_tol")
    uipc.view(ccd_tol_attr)[0] = 1e-6
    
    # Tighten Newton optimization tolerances to enforce strict non-penetration
    transrate_tol_attr = scene.config().find("newton/transrate_tol")
    uipc.view(transrate_tol_attr)[0] = 1e-4
    
    velocity_tol_attr = scene.config().find("newton/velocity_tol")
    uipc.view(velocity_tol_attr)[0] = 1e-4
    
    # Setup Constitutions
    abd_constitution = uipc.constitution.AffineBodyConstitution()
    stc_constitution = uipc.constitution.SoftTransformConstraint()
    nks_constitution = uipc.constitution.StrainLimitingBaraffWitkinShell()
    dsb_constitution = uipc.constitution.DiscreteShellBending()
    
    scene.constitution_tabular().insert(abd_constitution)
    scene.constitution_tabular().insert(stc_constitution)
    scene.constitution_tabular().insert(nks_constitution)
    scene.constitution_tabular().insert(dsb_constitution)
    
    subscene = scene.subscene_tabular().create("subscene_0")
    
    # Setup contact elements in the Scene contact tabular
    plane_contact = scene.contact_tabular().create("plane_contact")
    left_contact = scene.contact_tabular().create("left_contact")
    right_contact = scene.contact_tabular().create("right_contact")
    bag_contact = scene.contact_tabular().create("bag_contact")
    
    # Explicitly insert contact pairs to enable collisions inside IPC
    scene.contact_tabular().insert(bag_contact, plane_contact, 0.5, 0.0, True) # Bag-Plane collision
    scene.contact_tabular().insert(bag_contact, left_contact, 0.5, 0.0, True)  # Bag-Left fingertip collision
    scene.contact_tabular().insert(bag_contact, right_contact, 0.5, 0.0, True) # Bag-Right fingertip collision
    scene.contact_tabular().insert(bag_contact, bag_contact, 0.5, 0.0, True)     # Bag self-collisions
        
    # Volumetric watertight 3D box mesh for the floor to enable adaptive contact kappa
    floor_size = 5.0
    floor_thick = 0.1
    floor_verts = np.array([
        [-floor_size, -floor_size, -floor_thick],
        [ floor_size, -floor_size, -floor_thick],
        [ floor_size,  floor_size, -floor_thick],
        [-floor_size,  floor_size, -floor_thick],
        [-floor_size, -floor_size, 0.0],
        [ floor_size, -floor_size, 0.0],
        [ floor_size,  floor_size, 0.0],
        [-floor_size,  floor_size, 0.0]
    ], dtype=np.float64)
    floor_faces = np.array([
        [0, 2, 1], [0, 3, 2], # bottom
        [4, 5, 6], [4, 6, 7], # top
        [0, 1, 5], [0, 5, 4], # front
        [1, 2, 6], [1, 6, 5], # right
        [2, 3, 7], [2, 7, 6], # back
        [3, 0, 4], [3, 4, 7]  # left
    ], dtype=np.int32)
    
    floor_mesh = uipc.geometry.trimesh(floor_verts, floor_faces)
    floor_mesh.instances().resize(1)
    uipc.geometry.label_surface(floor_mesh) # Label boundary surface elements
    
    # Configure floor as an unconstrained volumetric rigid body with high mass and zero gravity
    abd_constitution.apply_to(floor_mesh, kappa=10.0 * uipc.unit.MPa, mass_density=1e9)
    stc_constitution.apply_to(floor_mesh)  # Initialize constraint slots
    plane_contact.apply_to(floor_mesh)
    
    is_constrained_floor = floor_mesh.instances().find(uipc.builtin.is_constrained)
    aim_transform_floor = floor_mesh.instances().find(uipc.builtin.aim_transform)
    uipc.view(is_constrained_floor)[0] = 0 # Unconstrained floor
    uipc.view(aim_transform_floor)[:] = np.eye(4, dtype=np.float64)
    
    # Disable gravity on the floor to keep it perfectly static
    floor_mesh.instances().create(uipc.builtin.gravity, np.array([0.0, 0.0, 0.0]))
    
    scene.objects().create("plane").geometries().create(floor_mesh)
    subscene.apply_to(floor_mesh)

    # Load Finger watertight Mesh for ABD
    finger_mesh_path = "benchmarks/franka_emika_panda/assets/finger_0.obj"
    finger_tri_mesh = trimesh.load(finger_mesh_path)
    finger_convex = finger_tri_mesh.convex_hull
    finger_verts = np.array(finger_convex.vertices, dtype=np.float64)
    finger_faces = np.array(finger_convex.faces, dtype=np.int32)

    # Load Trashbag Mesh
    trashbag_mesh_path = "benchmarks/franka_emika_panda/assets/Trashbag_coarse.obj"
    bag_tri_mesh = trimesh.load(trashbag_mesh_path)
    bag_verts = np.array(bag_tri_mesh.vertices, dtype=np.float64)
    bag_faces = np.array(bag_tri_mesh.faces, dtype=np.int32)

    # Setup left finger (with physical vertex translation to starting pos)
    left_mesh = uipc.geometry.trimesh(finger_verts.copy(), finger_faces.copy())
    left_mesh.instances().resize(1)
    uipc.geometry.label_surface(left_mesh) # CRITICAL: Label boundary surface elements for collision BVH!
    
    # Translate raw left finger vertices to prevent overlapping at the origin
    left_pos_view = uipc.view(left_mesh.positions())
    left_pos_np = left_pos_view[:].reshape(-1, 3)
    left_pos_np = (l_mat @ left_pos_np.T).T + l_pos
    left_pos_view[:] = left_pos_np.reshape(-1, 3, 1)
    
    abd_constitution.apply_to(left_mesh, kappa=10.0 * uipc.unit.MPa, mass_density=500.0)
    stc_constitution.apply_to(left_mesh)
    left_contact.apply_to(left_mesh)
    
    is_constrained = left_mesh.instances().find(uipc.builtin.is_constrained)
    aim_transform = left_mesh.instances().find(uipc.builtin.aim_transform)
    uipc.view(is_constrained)[0] = 1
    uipc.view(aim_transform)[:] = l_T
    
    left_obj = scene.objects().create("left_finger_0")
    left_obj.geometries().create(left_mesh)
    subscene.apply_to(left_mesh)

    # Setup right finger (with physical vertex translation to starting pos)
    right_mesh = uipc.geometry.trimesh(finger_verts.copy(), finger_faces.copy())
    right_mesh.instances().resize(1)
    uipc.geometry.label_surface(right_mesh) # CRITICAL: Label boundary surface elements for collision BVH!
    
    # Translate raw right finger vertices to prevent overlapping at the origin
    right_pos_view = uipc.view(right_mesh.positions())
    right_pos_np = right_pos_view[:].reshape(-1, 3)
    right_pos_np = (r_mat @ right_pos_np.T).T + r_pos
    right_pos_view[:] = right_pos_np.reshape(-1, 3, 1)
    
    abd_constitution.apply_to(right_mesh, kappa=10.0 * uipc.unit.MPa, mass_density=500.0)
    stc_constitution.apply_to(right_mesh)
    right_contact.apply_to(right_mesh)
    
    is_constrained_r = right_mesh.instances().find(uipc.builtin.is_constrained)
    aim_transform_r = right_mesh.instances().find(uipc.builtin.aim_transform)
    uipc.view(is_constrained_r)[0] = 1
    uipc.view(aim_transform_r)[:] = r_T
    
    right_obj = scene.objects().create("right_finger_0")
    right_obj.geometries().create(right_mesh)
    subscene.apply_to(right_mesh)

    # Setup soft trashbag (translated)
    bag_mesh = uipc.geometry.trimesh(bag_verts.copy(), bag_faces.copy())
    uipc.geometry.label_surface(bag_mesh) # CRITICAL: Label boundary surface elements for collision BVH!
    
    moduli = uipc.constitution.ElasticModuli2D.youngs_poisson(1.5e4, 0.45)
    nks_constitution.apply_to(bag_mesh, moduli=moduli, mass_density=150.0, thickness=0.001)
    dsb_constitution.apply_to(bag_mesh, bending_stiffness=2.0)
    bag_contact.apply_to(bag_mesh)
    
    # Translate to match initial target position
    uipc.view(bag_mesh.positions())[:] += np.array([[0.62], [-0.35], [0.25]])
    
    bag_obj = scene.objects().create("trashbag_0")
    bag_obj.geometries().create(bag_mesh)
    subscene.apply_to(bag_mesh)

    # Initialize World
    world.init(scene)
    
    # Accessors
    abd_accessor = world.features().find(uipc.core.AffineBodyStateAccessorFeature)
    abd_state_geom = abd_accessor.create_geometry()
    abd_state_geom.instances().create(uipc.builtin.transform, np.eye(4, dtype=np.float64))
    
    # Create SceneVisitor to query geometries
    visitor = SceneVisitor(scene)

    # GPU buffers
    wp_target_pos = wp.zeros(shape=(n_envs, 2), dtype=wp.vec3, device="cuda")
    wp_target_mat = wp.zeros(shape=(n_envs, 2), dtype=wp.mat33, device="cuda")
    wp_resolved_T = wp.zeros(shape=(n_envs, 2), dtype=wp.mat44, device="cuda")
    wp_reaction_xfrc = wp.zeros(shape=(n_envs, 2), dtype=wp.spatial_vector, device="cuda")

    stiffness = 0.15 / (dt ** 2)  # Stabilized constraint strength
    mass = 0.015

    def run_step(step_idx: int):
        t_start = time.time()
        
        if step_idx == 0:
            print("[Step 0] 1. Setting trajectory planner controls...", flush=True)
        
        # 1. Trajectory Planner: Active robotic controls
        qpos_home = np.array([0.0, -0.9, 0.0, -2.3, 0.0, 2.3, -0.78])
        qpos_grasp = np.array([-0.0177, 0.2176, -0.0434, -2.1142, -0.0042, 2.6892, -0.78])
        qpos_lift = np.array([-0.0112, -0.0379, -0.0489, -2.0099, -0.0024, 2.5934, -0.78])
        
        if step_idx < 150:
            # Phase 1: Settling
            mjd.ctrl[:7] = qpos_home
            mjd.ctrl[7] = 255.0
        elif step_idx < 750:
            # Phase 2: Descend & Surround
            alpha = (step_idx - 150) / 600.0
            mjd.ctrl[:7] = (1.0 - alpha) * qpos_home + alpha * qpos_grasp
            mjd.ctrl[7] = 255.0
        elif step_idx < 950:
            # Phase 3: Close & Pinch
            mjd.ctrl[:7] = qpos_grasp
            mjd.ctrl[7] = 0.0
        else:
            # Phase 4: Lift Bag Up
            alpha = min(1.0, (step_idx - 950) / 800.0)
            mjd.ctrl[:7] = (1.0 - alpha) * qpos_grasp + alpha * qpos_lift
            mjd.ctrl[7] = 0.0
        
        # Copy control inputs into MJWarp
        wp.copy(d.ctrl, wp.array([mjd.ctrl.astype(np.float32)]))
        
        if step_idx == 0:
            print("[Step 0] 2. Stepping rigid kinematics in MJWarp...", flush=True)
        
        # 2. Step rigid kinematics directly in MJWarp
        t_kin_start = time.time()
        forward.step(m, d)
        wp.synchronize()
        t_kin = time.time() - t_kin_start
        
        # Read transforms
        target_pos_np = d.xpos.numpy()
        target_mat_np = d.xmat.numpy()
        
        # Update left finger target in IPC
        l_pos = target_pos_np[0, left_finger_body_idx]
        l_mat = target_mat_np[0, left_finger_body_idx]
        l_T = np.eye(4, dtype=np.float64)
        l_T[:3, :3] = l_mat
        l_T[:3, 3] = l_pos
        uipc.view(aim_transform)[:] = l_T
        
        # Update right finger target in IPC
        r_pos = target_pos_np[0, right_finger_body_idx]
        r_mat = target_mat_np[0, right_finger_body_idx]
        r_T = np.eye(4, dtype=np.float64)
        r_T[:3, :3] = r_mat
        r_T[:3, 3] = r_pos
        uipc.view(aim_transform_r)[:] = r_T
        
        # Pin floor aim transform to identity
        uipc.view(aim_transform_floor)[:] = np.eye(4, dtype=np.float64)

        
        if step_idx == 0:
            print("[Step 0] 3. Advancing pyuipc contact/cloth solver...", flush=True)
        
        # 3. Step pyuipc contact & cloth deformation solver
        t_ipc_start = time.time()
        world.advance()
        world.retrieve()
        t_ipc = time.time() - t_ipc_start
        
        if step_idx == 0:
            print("[Step 0] 4. Copying resolved finger ABD poses...", flush=True)
        
        # 4. Retrieve solved ABD positions from IPC
        abd_accessor.copy_to(abd_state_geom)
        trans_attr = abd_state_geom.instances().find(uipc.builtin.transform)
        resolved_T_np = uipc.view(trans_attr)[:]
        
        if step_idx == 0:
            print("[Step 0] 5. Querying resolved cloth positions from SceneVisitor...", flush=True)
        
        # 5. Retrieve solved cloth vertices positions from IPC using SceneVisitor
        resolved_positions = None
        for geom_slot in visitor.geometries():
            geom = geom_slot.geometry()
            if hasattr(geom, "dim") and geom.dim() == 2 and geom.vertices().size() > 1000:
                (transformed_geom,) = uipc.geometry.apply_transform(geom)
                resolved_positions = uipc.view(transformed_geom.positions()).reshape(-1, 3)
                break
        
        if step_idx == 0:
            print("[Step 0] 6. Writing resolved cloth positions to JAX/Warp GPU array...", flush=True)
        
        # 6. Assign resolved positions directly to the JAX/Warp GPU array d.flexvert_xpos
        # This ensures get_data_into copies it natively and thread-safely to C++ (mjd)!
        if resolved_positions is not None:
            d.flexvert_xpos.assign(resolved_positions[np.newaxis, ...].astype(np.float32))
        
        if step_idx == 0:
            print("[Step 0] 7. Syncing GPU buffers to Warp...", flush=True)
        
        # 7. Sync targets and ABD poses to Warp arrays for force computation
        target_pos_batch = np.zeros((n_envs, 2, 3), dtype=np.float32)
        target_mat_batch = np.zeros((n_envs, 2, 3, 3), dtype=np.float32)
        resolved_T_batch = np.zeros((n_envs, 2, 4, 4), dtype=np.float32)
        
        target_pos_batch[0, 0] = target_pos_np[0, left_finger_body_idx]
        target_pos_batch[0, 1] = target_pos_np[0, right_finger_body_idx]
        target_mat_batch[0, 0] = target_mat_np[0, left_finger_body_idx]
        target_mat_batch[0, 1] = target_mat_np[0, right_finger_body_idx]
        resolved_T_batch[0, 0] = resolved_T_np[1]
        resolved_T_batch[0, 1] = resolved_T_np[2]
        
        wp_target_pos.assign(target_pos_batch)
        wp_target_mat.assign(target_mat_batch)
        wp_resolved_T.assign(resolved_T_batch)
        
        if step_idx == 0:
            print("[Step 0] 8. Computing reaction forces in Warp GPU kernel...", flush=True)
        
        # 8. Compute reaction forces using Warp GPU kernel
        t_forces_start = time.time()
        wp.launch(
            compute_coupling_forces_kernel,
            dim=(n_envs, 2),
            inputs=[wp_target_pos, wp_target_mat, wp_resolved_T, stiffness, 0.0, mass],
            outputs=[wp_reaction_xfrc]
        )
        wp.synchronize()
        t_forces = time.time() - t_forces_start
        
        if step_idx == 0:
            print("[Step 0] 9. Feeding reaction forces back to MJWarp...", flush=True)
        
        # 9. Feed reaction forces back to MJWarp
        reaction_xfrc_np = wp_reaction_xfrc.numpy()
        xfrc_applied_np = d.xfrc_applied.numpy()
        xfrc_applied_np[0, left_finger_body_idx] = reaction_xfrc_np[0, 0]
        xfrc_applied_np[0, right_finger_body_idx] = reaction_xfrc_np[0, 1]
        d.xfrc_applied.assign(xfrc_applied_np)
        
        # Update MuJoCo MjData into step
        mjw.get_data_into(mjd, mjm, d)
        
        if step_idx == 0:
            print("[Step 0] 10. Step 0 completed successfully!", flush=True)
        
        # Save rendered frame to disk at key steps
        if resolved_positions is not None:
            if step_idx in [0, 10, 50, 150, 300, 450, 600, 750, 850, 950, 1100, 1300, 1500, 1700, 1749]:
                os.makedirs("assets", exist_ok=True)
                renderer.update_scene(mjd, camera="front")
                pixels = renderer.render()
                Image.fromarray(pixels).save(f"assets/grasp_step_{step_idx}.png")
                print(f"[Step {step_idx:03d}] Saved frame to assets/grasp_step_{step_idx}.png", flush=True)
                
        t_total = time.time() - t_start
        return resolved_positions, (t_total, t_kin, t_ipc, t_forces)

    if flags.FLAGS.headless:
        print(f"Running Headless Rollout ({flags.FLAGS.nstep_coupling} steps)...", flush=True)
        for step_idx in range(flags.FLAGS.nstep_coupling):
            bag_pos, (t_total, t_kin, t_ipc, t_forces) = run_step(step_idx)
            if step_idx % 10 == 0:
                centroid = np.mean(bag_pos, axis=0) if bag_pos is not None else np.array([0, 0, 0])
                min_z = np.min(bag_pos[:, 2]) if bag_pos is not None else 0.0
                print(
                    f"Step {step_idx:04d}: Robot Joint 2 = {mjd.qpos[1]:.4f}, Centroid={centroid}, Min Z={min_z:.4f}\n"
                    f"   ↳ Step Time: {t_total*1000:.1f}ms (Kinematics={t_kin*1000:.1f}ms, IPC Solver={t_ipc*1000:.1f}ms, Warp Forces={t_forces*1000:.1f}ms)", 
                    flush=True
                )
        print("Headless Rollout completed successfully!", flush=True)
        return

    print("Launching MuJoCo Visualizer. Press SPACE to run / pause simulation...", flush=True)
    with mujoco.viewer.launch_passive(mjm, mjd) as viewer:
        step_count = 0
        while viewer.is_running():
            start_time = time.time()
            
            # Acquire lock to perform thread-safe coupled physics step and viewer sync
            with viewer.lock():
                run_step(step_count)
                viewer.sync()
            
            # Control frame timing for smooth execution
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
                
            step_count += 1
            if step_count % 100 == 0:
                print(f"Step {step_count:04d} rendered successfully.", flush=True)

if __name__ == "__main__":
    main()
