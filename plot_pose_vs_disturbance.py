"""
plot_pose_vs_disturbance.py
============================
Stick-figure pose comparison at increasing perturbation levels, in the
style of Fig. 8 (paper reference). Draws the actual model geometry (foot
pivot -> leg -> trunk) at the joint angles reached during rollouts at
several force magnitudes, using real qpos values, not illustration.

Open this in VS Code and edit freely -- FORCE_LEVELS, which candidate,
which joint to report, and the drawing geometry are all plain variables
below.

Usage:
    python plot_pose_vs_disturbance.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from candidate2_yaxis import Candidate2Env   # swap candidate here if needed

MODEL = "ppo_candidate2_yaxis_015_107"       # confirm this matches your final model
VECNORM = "vecnormalize_candidate2_yaxis_015_107.pkl"

# Your actual tested range, not the paper's numbers -- force_range=(0,30)
# in candidate2_yaxis.py is the real ceiling.
FORCE_LEVELS = [0, 10, 20, 30]

# Leg segment lengths -- must match the XML capsule geometry
# (leg_geom fromto="0 0 0 0 0 0.9", trunk_geom fromto="0 0 0 0 0 0.6")
LEG_LEN = 0.9
TRUNK_LEN = 0.6


def rollout_at_force(force):
    """Runs one episode with the perturbation forced on every step at a
    fixed magnitude (rather than the trained stochastic disturb_prob),
    to get a clean 'maximum lean under this exact force' reading."""
    def make_env():
        e = Candidate2Env(disturb_prob=1.0, force_range=(force, force))
        return TimeLimit(e, max_episode_steps=1000)

    venv = DummyVecEnv([make_env])
    venv = VecNormalize.load(VECNORM, venv)
    venv.training = False
    venv.norm_reward = False
    venv.seed(0)
    obs = venv.reset()

    model = PPO.load(MODEL)
    max_lean_qpos = None
    max_lean_angle = -np.inf

    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, d, infos = venv.step(a)
        done = bool(d[0])
        raw_env = venv.venv.envs[0].unwrapped
        qpos = raw_env.data.qpos[:4].copy()
        ankle_flexion = qpos[1]   # index 1 = ankle_flexion, AP-plane, per JOINT_NAMES
        if ankle_flexion > max_lean_angle:
            max_lean_angle = ankle_flexion
            max_lean_qpos = qpos.copy()
    venv.close()
    return max_lean_qpos, np.degrees(max_lean_angle)


def draw_pose(ax, qpos, title, baseline_angle=None):
    """qpos = [ankle_eversion, ankle_flexion, hip_abduction, hip_flexion]
    Draws in the AP plane (ankle_flexion, hip_flexion), matching the
    reference figure's sagittal-view stick figure."""
    ankle_flex = qpos[1]
    hip_flex = qpos[3]

    ax.plot([-0.15, 0.15], [0, 0], color="0.3", linewidth=4, solid_capstyle="round")

    leg_x = LEG_LEN * np.sin(ankle_flex)
    leg_y = LEG_LEN * np.cos(ankle_flex)
    ax.plot([0, leg_x], [0, leg_y], color="#0072B2", linewidth=6, solid_capstyle="round")

    trunk_angle = ankle_flex + hip_flex  # dropped the "- pi/2" term
    # NOTE: verify this offset against your XML's joint zero-reference --
    # adjust the -pi/2 term if the trunk appears rotated wrong.
    trunk_x = leg_x + TRUNK_LEN * np.sin(trunk_angle)
    trunk_y = leg_y + TRUNK_LEN * np.cos(trunk_angle)
    ax.plot([leg_x, trunk_x], [leg_y, trunk_y], color="#D55E00", linewidth=6, solid_capstyle="round")

    theta_deg = np.degrees(ankle_flex)
    arc = np.linspace(0, ankle_flex, 30)
    ax.plot(0.12 * np.sin(arc), 0.12 * np.cos(arc), color="0.4", linewidth=1)
    ax.text(0.18, 0.06, f"theta={theta_deg:.1f} deg", fontsize=9, ha="left")

    if baseline_angle is not None:
        delta = theta_deg - baseline_angle
        ax.text(0.18, -0.02, f"delta={delta:+.1f} deg", fontsize=9, ha="left", color="0.3")

    ax.set_xlim(-0.5, 0.9)
    ax.set_ylim(-0.1, 1.7)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


if __name__ == "__main__":
    fig, axes = plt.subplots(1, len(FORCE_LEVELS), figsize=(4 * len(FORCE_LEVELS), 4.5))

    baseline_angle = None
    for i, force in enumerate(FORCE_LEVELS):
        qpos, angle_deg = rollout_at_force(force)
        print(f"force={force}N: ankle_flexion={np.degrees(qpos[1]):.1f}deg, hip_flexion={np.degrees(qpos[3]):.1f}deg")
        if force == 0:
            baseline_angle = angle_deg
            title = "No disturbance"
        else:
            title = f"{force}N (max sustained lean)"
        draw_pose(axes[i], qpos, title, baseline_angle=baseline_angle if force != 0 else None)
        print(f"force={force}N: ankle_flexion max = {angle_deg:.2f} deg")

    fig.suptitle("Maximum ankle-flexion angle vs. perturbation magnitude", fontsize=12)
    plt.tight_layout()
    plt.savefig("pose_vs_disturbance.png", dpi=200, bbox_inches="tight")
    print("\nSaved pose_vs_disturbance.png")
