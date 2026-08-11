"""
plot_com_xy_fixed.py
=======================
ANSWERS: why do com_xy plots show all starting points near y=0 (with big
x variability, -0.1 to 0.1) and all ending points around y=0.25 (still
variable)? (Supervisor question.)

DIAGNOSIS: this is the signature of an axis-referencing mismatch -- com_x
is being zero-referenced (each trial's own start subtracted, matching the
convention used everywhere ELSE in this project) but com_y is NOT, so
com_y plots its RAW value, which sits near a fixed nonzero offset (~0.25)
determined by the model's default pose/geometry, not by trial-to-trial
behavior. The "start near 0, drift to ~0.25" pattern is consistent with
com_y beginning near its (near-zero) INITIAL simulation value and then
settling toward its STEADY-STATE raw value once the episode's dynamics
settle -- not a real behavioral signal.

THE FIX: zero-reference BOTH axes (each relative to ITS OWN value at
step 0), exactly as already done for com_x throughout this project's
other comparison scripts. This makes the com_xy plot show DISPLACEMENT
FROM START in both dimensions, which is the correct, comparable quantity,
consistent with every other plot in this project.

Usage:
    python plot_com_xy_fixed.py <candidate_key>
Edit CONFIGS to match your existing candidate module/model naming.
"""
import sys
import numpy as np
import plotly.graph_objects as go

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

N_EPISODES = 20

CONFIGS = {
        "candidate1": dict(module="candidate1_ap_comy", cls="Candidate1Env",
                                        model="ppo_candidate1_ap_comy1_staypenalty6",
                                        vecnorm="vecnormalize_candidate1_ap_comy1_staypenalty6.pkl"),
        "candidate2": dict(module="candidate2_ap_comy1_staypenalty_jointfix", cls="Candidate2Env",
                                        model="ppo_candidate2_ap_comy1_staypenalty_jointfix",
                                        vecnorm="vecnormalize_candidate2_ap_comy1_staypenalty_jointfix.pkl"),
        "candidate3": dict(module="candidate3_ap_comy1_staypenalty", cls="Candidate3Env",
                                        model="ppo_candidate3_ap_comy1_staypenalty6",
                                        vecnorm="vecnormalize_candidate3_ap_comy1_staypenalty6.pkl"),
    }


def collect_com_xy(cfg, n_episodes=N_EPISODES):
    module = __import__(cfg["module"])
    EnvClass = getattr(module, cfg["cls"])
    model = PPO.load(cfg["model"])

    all_x_raw, all_y_raw = [], []
    for ep in range(n_episodes):
        target = 0.08 if ep % 2 == 0 else -0.08
        venv = DummyVecEnv([lambda: TimeLimit(EnvClass(fixed_target=target), max_episode_steps=1000)])
        venv = VecNormalize.load(cfg["vecnorm"], venv)
        venv.training = False
        venv.norm_reward = False
        venv.seed(ep)
        obs = venv.reset()

        x_traj, y_traj = [], []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done_v, info = venv.step(action)
            done = bool(done_v[0])
            x_traj.append(info[0]["com_x"])
            y_traj.append(info[0]["com_y"])
        venv.close()
        all_x_raw.append(np.array(x_traj))
        all_y_raw.append(np.array(y_traj))
    return all_x_raw, all_y_raw


if __name__ == "__main__":
    cfg_key = sys.argv[1] if len(sys.argv) > 1 else "candidate2"
    cfg = CONFIGS[cfg_key]

    print(f"Rolling out {cfg_key}...")
    all_x_raw, all_y_raw = collect_com_xy(cfg)

    # DIAGNOSTIC PRINT: confirms the bug before fixing it. If raw com_y at
    # step 0 is NOT near 0, and raw com_y later settles near a fixed value
    # regardless of trial, that confirms the "unreferenced axis" diagnosis.
    y_starts = [y[0] for y in all_y_raw]
    y_ends = [y[-1] for y in all_y_raw]
    print(f"Raw com_y at step 0: mean={np.mean(y_starts):.4f}, std={np.std(y_starts):.4f}")
    print(f"Raw com_y at final step: mean={np.mean(y_ends):.4f}, std={np.std(y_ends):.4f}")
    print("(If start is near 0 and end clusters near ~0.25 regardless of target,")
    print(" this confirms com_y was plotted RAW/unreferenced in the original figure.)\n")

    # THE FIX: zero-reference BOTH axes, same convention as com_x everywhere else.
    fig_raw = go.Figure()
    fig_fixed = go.Figure()

    for i, (x_raw, y_raw) in enumerate(zip(all_x_raw, all_y_raw)):
        color = "royalblue" if i % 2 == 0 else "firebrick"
        # RAW (reproduces the original, "messy" plot for comparison)
        fig_raw.add_trace(go.Scatter(x=x_raw, y=y_raw, mode="lines",
                                       line=dict(color=color, width=1), opacity=0.5,
                                       showlegend=False))
        # FIXED: zero-reference both x and y to their own trial start
        x_fixed = x_raw - x_raw[0]
        y_fixed = y_raw - y_raw[0]
        fig_fixed.add_trace(go.Scatter(x=x_fixed, y=y_fixed, mode="lines",
                                         line=dict(color=color, width=1), opacity=0.5,
                                         showlegend=False))

    fig_raw.update_layout(title=f"BEFORE (raw, unreferenced): {cfg_key} com_x vs com_y",
                           xaxis_title="com_x (raw, m)", yaxis_title="com_y (raw, m)",
                           template="plotly_white", width=700, height=600)
    fig_fixed.update_layout(title=f"AFTER (zero-referenced, both axes): {cfg_key} com_x vs com_y",
                             xaxis_title="com_x displacement from start (m)",
                             yaxis_title="com_y displacement from start (m)",
                             template="plotly_white", width=700, height=600)

    fig_raw.write_html(f"com_xy_raw_{cfg_key}.html")
    fig_fixed.write_html(f"com_xy_fixed_{cfg_key}.html")
    print(f"Saved com_xy_raw_{cfg_key}.html (reproduces the original messy plot)")
    print(f"Saved com_xy_fixed_{cfg_key}.html (corrected: both axes zero-referenced)")
