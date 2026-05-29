"""Vanilla MuJoCo Flex Rollout Verification Script.

Loads the vanilla scene with native soft-body flex, steps it natively for 400 steps,
and saves visual PNG frames from the 'front' camera to assets/vanilla_step_*.png.
"""

import os
import numpy as np
from PIL import Image
import mujoco

def main():
    os.makedirs("assets", exist_ok=True)
    
    print("Loading vanilla MuJoCo scene...")
    xml_path = "benchmarks/franka_emika_panda/panda_trashbag_vanilla.xml"
    mjm = mujoco.MjModel.from_xml_path(xml_path)
    mjd = mujoco.MjData(mjm)
    
    # 1. Set starting posture for the robot (stable arm posture)
    # Franka has 9 joints (7 arm joints, 2 finger slide joints)
    qpos_start = np.array([0.0, -0.9, 0.0, -2.3, 0.0, 2.3, -0.78, 0.04, 0.04])
    mjd.qpos[:9] = qpos_start
    # Set active position control target to keep the robot fixed
    mjd.ctrl[:8] = [0.0, -0.9, 0.0, -2.3, 0.0, 2.3, -0.78, 255.0] # gripper open
    
    # Initialize Renderer
    renderer = mujoco.Renderer(mjm, height=480, width=640)
    
    n_steps = 400
    save_steps = [0, 50, 100, 200, 300, 400]
    
    print(f"Starting vanilla rollout for {n_steps} steps...")
    print("-" * 60)
    
    for step in range(n_steps + 1):
        # Synchronize kinematics
        mujoco.mj_kinematics(mjm, mjd)
        
        # Render and save at milestones
        if step in save_steps:
            print(f"[Step {step:03d}] Rendering visual frame...")
            renderer.update_scene(mjd, camera="front")
            pixels = renderer.render()
            
            # Save as PNG
            img = Image.fromarray(pixels)
            filename = f"assets/vanilla_step_{step}.png"
            img.save(filename)
            print(f"Saved visual frame to {filename}")
            
        # Advance native physics step (integrator, CG solver, collisions, etc.)
        if step < n_steps:
            mujoco.mj_step(mjm, mjd)
            
    print("-" * 60)
    print("Vanilla verification completed successfully!")
    print("All assets/vanilla_step_*.png frames saved.")

if __name__ == "__main__":
    main()
