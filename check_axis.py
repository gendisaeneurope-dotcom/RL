"""Confirms which physical plane each CoM axis corresponds to, by
displacing one joint at a time and observing which axis moves.

ankle_eversion and hip_abduction are FRONTAL-plane (mediolateral) joints.
ankle_flexion and hip_flexion are SAGITTAL-plane (anterior-posterior).

If com_x responds to the flexion joints, com_x is the AP axis.
If com_x responds to the eversion/abduction joints, com_x is the ML axis.
"""
import numpy as np
import mujoco
from candidate2_ap_comy1_staypenalty import Candidate2Env, JOINT_NAMES

env = Candidate2Env()
env.reset(seed=0)

base_x, base_y = env._com_xy()
print(f"baseline: com_x={base_x:.5f}, com_y={base_y:.5f}\n")

for j, name in enumerate(JOINT_NAMES):
    env.reset(seed=0)
    env.data.qpos[:4] = 0.0
    mujoco.mj_forward(env.model, env.data)
    x0, y0 = env._com_xy()

    env.data.qpos[j] = np.radians(15.0)   # displace this joint only
    mujoco.mj_forward(env.model, env.data)
    x1, y1 = env._com_xy()

    dx, dy = x1 - x0, y1 - y0
    dominant = "com_x" if abs(dx) > abs(dy) else "com_y"
    print(f"{name:18s}  dx={dx:+.5f}  dy={dy:+.5f}   -> moves {dominant}")