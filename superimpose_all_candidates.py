"""
superimpose_all_candidates.py
===============================
THE "SUPERIMPOSE" PLOT: all three candidates' mean CoM-x trajectories
PLUS the human mean trajectory, overlaid on ONE shared axis.

WHY THIS DIFFERS FROM compare_sim_vs_human_rmse_v3.py:
That script makes ONE figure per candidate (sim vs. human, one pair at a
time) -- useful for detailed per-candidate diagnostics, but it does not
let you visually compare candidates AGAINST EACH OTHER in a single glance.
This script produces the single combined figure your supervisor will most
likely want to see first: "here are all three reward designs, and here is
the human reference, on the same plot" -- directly analogous to comparison
figures common in this literature (e.g. multiple biomechanical models
overlaid against experimental ground truth on one axis).

WHAT IT PRODUCES
-----------------
ONE figure: normalized time (x-axis) vs. zero-referenced, direction-
canonicalized CoM-x displacement (y-axis), with FOUR lines:
  - Human (red, thick, drawn on top)
  - Candidate 1 (blue)
  - Candidate 2 (green)
  - Candidate 3 (purple)
Each sim line is a mean over its 40 evaluation episodes; shaded bands are
omitted here deliberately (four overlapping shaded regions is visually
unreadable) -- use compare_sim_vs_human_rmse_v3.py's per-candidate figures
if you need to see variance bands for a specific candidate.

Requires the three FINAL (stay-penalty) trained models to already exist:
    ppo_candidate1_ap_comy1_staypenalty6 / vecnormalize_..._staypenalty6.pkl
    ppo_candidate2_ap_comy1_staypenalty_6 / vecnormalize_..._staypenalty_6.pkl
    ppo_candidate3_ap_comy1_staypenalty6 / vecnormalize_..._staypenalty6.pkl

Usage:
    python superimpose_all_candidates.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

HUMAN_CSV_PATH = "C:\\Gepi10\\SPACEMED\\Sem4_Thesis\\Thesis\\Workspace\\Gymnasium_RL\\human_com_cleaned_subject003_v7.csv"
HUMAN_COMX_COL = "com_x_human"
HUMAN_TRIAL_COL = "trial_id"
RESAMPLE_LEN = 60
N_SIM_EPISODES = 40

CONFIGS = {
    "candidate1": dict(module="candidate1_ap_comy", cls="Candidate1Env",
                        model="ppo_candidate1_ap_comy1_staypenalty6",
                        vecnorm="vecnormalize_candidate1_ap_comy1_staypenalty6.pkl",
                         color="#1F77B4"),
    "candidate2": dict(module="candidate2_ap_comy1_staypenalty_jointfix", cls="Candidate2Env",
                        model="ppo_candidate2_ap_comy1_staypenalty_jointfix",
                        vecnorm="vecnormalize_candidate2_ap_comy1_staypenalty_jointfix.pkl",
                         color="#2CA02C"),
    "candidate3": dict(module="candidate3_ap_comy1_staypenalty", cls="Candidate3Env",
                        model="ppo_candidate3_ap_comy1_staypenalty6",
                        vecnorm="vecnormalize_candidate3_ap_comy1_staypenalty6.pkl",
                         color="#9467BD"),
}
HUMAN_COLOR = "#D62728"


def resample_to_fixed_length(traj, length=RESAMPLE_LEN):
    traj = np.asarray(traj, dtype=float)
    old_x = np.linspace(0, 1, len(traj))
    new_x = np.linspace(0, 1, length)
    return np.interp(new_x, old_x, traj)


def zero_reference(traj):
    traj = np.asarray(traj, dtype=float)
    return traj - traj[0]


def canonicalize_direction(traj_zeroed):
    net_direction = np.sign(traj_zeroed[-1] - traj_zeroed[0])
    if net_direction > 0:
        traj_zeroed = -traj_zeroed
    return traj_zeroed


def load_human_mean():
    df = pd.read_csv(HUMAN_CSV_PATH, low_memory=False)
    trajs = []
    for trial_id, g in df.groupby(HUMAN_TRIAL_COL):
        vals = g[HUMAN_COMX_COL].to_numpy()
        if len(vals) < 5:
            continue
        vals_canon = canonicalize_direction(zero_reference(vals))
        trajs.append(resample_to_fixed_length(vals_canon))
    trajs = np.array(trajs)
    return trajs.mean(axis=0), trajs.std(axis=0), len(trajs)


def collect_candidate_mean(cfg, n_episodes=N_SIM_EPISODES):
    module = __import__(cfg["module"])
    EnvClass = getattr(module, cfg["cls"])
    model = PPO.load(cfg["model"])

    trajs = []
    for ep in range(n_episodes):
        target = 0.08 if ep % 2 == 0 else -0.08
        venv = DummyVecEnv([lambda: TimeLimit(EnvClass(fixed_target=target), max_episode_steps=1000)])
        venv = VecNormalize.load(cfg["vecnorm"], venv)
        venv.training = False
        venv.norm_reward = False
        venv.seed(ep)
        obs = venv.reset()

        com_x_traj = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done_v, info = venv.step(action)
            done = bool(done_v[0])
            com_x_traj.append(float(info[0]["com_x"]))
        venv.close()

        traj_canon = canonicalize_direction(zero_reference(com_x_traj))
        trajs.append(resample_to_fixed_length(traj_canon))
    trajs = np.array(trajs)
    return trajs.mean(axis=0), trajs.std(axis=0)


if __name__ == "__main__":
    x = np.linspace(0, 100, RESAMPLE_LEN)

    print("Loading human reference data...")
    human_mean, human_std, n_human = load_human_mean()

    candidate_means = {}
    for name, cfg in CONFIGS.items():
        print(f"Rolling out {name}...")
        mean, std = collect_candidate_mean(cfg)
        candidate_means[name] = (mean, std)

    fig = go.Figure()

    # human drawn first (background), with a light shaded band, then each
    # candidate as a clean line on top -- ordering matters for legibility
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([human_mean + human_std, (human_mean - human_std)[::-1]]),
        fill="toself", fillcolor="rgba(214,39,40,0.12)", line=dict(width=0),
        showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=x, y=human_mean, mode="lines", name=f"Human (n={n_human})",
        line=dict(color=HUMAN_COLOR, width=4, dash="solid")))

    for name, cfg in CONFIGS.items():
        mean, _ = candidate_means[name]
        fig.add_trace(go.Scatter(
            x=x, y=mean, mode="lines", name=name,
            line=dict(color=cfg["color"], width=2.5)))

    fig.update_layout(
        title="Superimposed Comparison: All Reward Candidates vs. Human Reference<br>"
              "<sup>Zero-referenced, direction-canonicalized mean CoM-x trajectories</sup>",
        xaxis_title="Normalized trial time (%)",
        yaxis_title="CoM-x displacement from trial start (m)",
        template="plotly_white",
        font=dict(size=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0),
        width=1100, height=700,
    )
    fig.write_html("superimposed_all_candidates.html")
    try:
        fig.write_image("superimposed_all_candidates.png", scale=2)
    except Exception as e:
        print(f"(PNG export skipped -- run `pip install -U kaleido`: {e})")
    print("Saved superimposed_all_candidates.html (+ .png if kaleido available)")
