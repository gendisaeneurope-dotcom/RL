"""
superimpose_with_timing_fix.py
=================================
SAME SCRIPT AS superimpose_all_candidates.py, with ONE addition: the new
"Timing Fix" model (ppo_irl_timing_fix / vecnormalize_irl_timing_fix.pkl),
trained by add_timing_feature_and_retrain.py earlier -- NO NEW TRAINING
HAPPENS IN THIS SCRIPT. It only loads 4 ALREADY-TRAINED models and plots
their rollouts. If any of the 4 files listed in CONFIGS below don't exist
on your machine, that specific candidate will be skipped with a clear
message (not silently ignored).

WHY RUN THIS: your diagnosis (diagnose_stayed_then_moved_zero.py)
confirmed all 3 original candidates fail the "wait then move" pattern.
add_timing_feature_and_retrain.py trained ONE new policy specifically
targeting that gap. This script is the direct visual answer to "did that
targeted fix work" -- nothing more.

Usage:
    python superimpose_with_timing_fix.py
"""
import os
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
    # THE NEW ADDITION -- trained specifically to target the timing gap.
    "Timing Fix (new)": dict(module="add_timing_feature_and_retrain", cls="IRLCandidateEnv",
                          model="ppo_irl_timing_fix_v3_post5_early6",
                          vecnorm="vecnormalize_irl_timing_fix_v3_post5_early6.pkl",
                          color="#FF7F0E"),
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
        # "Timing Fix" env needs reward_weights to construct; use the SAME
        # residual weights the training script computed and printed, so
        # the rollout env matches what the policy was actually trained
        # under (only affects the reward signal, not the trained actions).
        if cfg["cls"] == "IRLCandidateEnv":
            reward_weights = np.array([0.0003, 0.0012, -0.0002, -0.0, -0.0068, 3.2785])
            env_instance = EnvClass(reward_weights=reward_weights, fixed_target=target)
        else:
            env_instance = EnvClass(fixed_target=target)

        venv = DummyVecEnv([lambda: TimeLimit(env_instance, max_episode_steps=1000)])
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
        model_file = cfg["model"] + ".zip"
        if not os.path.exists(model_file):
            print(f"SKIPPING {name}: model file {model_file} not found.")
            continue
        print(f"Rolling out {name} (loading existing trained model, no training)...")
        mean, std = collect_candidate_mean(cfg)
        candidate_means[name] = mean

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([human_mean + human_std, (human_mean - human_std)[::-1]]),
        fill="toself", fillcolor="rgba(214,39,40,0.12)", line=dict(width=0),
        showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=x, y=human_mean, mode="lines", name=f"Human (n={n_human})",
        line=dict(color=HUMAN_COLOR, width=4)))

    for name in candidate_means:
        fig.add_trace(go.Scatter(
            x=x, y=candidate_means[name], mode="lines", name=name,
            line=dict(color=CONFIGS[name]["color"], width=2.5)))

    fig.update_layout(
        title=dict(
            text="Does the Timing-Fix Model Reproduce Human Movement Shape?"
                 "<br><span style='font-size:13px;color:gray'>"
                 "Zero-referenced, direction-canonicalized mean CoM-x trajectories</span>",
            x=0.02, xanchor="left", y=0.97, yanchor="top",
        ),
        xaxis_title="Normalized trial time (%)",
        yaxis_title="CoM-x displacement from trial start (m)",
        template="plotly_white", font=dict(size=15),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        margin=dict(t=110, b=140, l=80, r=40),
        width=1100, height=750,
    )
    fig.write_html("superimposed_with_timing_fix.html")
    try:
        fig.write_image("superimposed_with_timing_fix.png", scale=2)
    except Exception as e:
        print(f"(PNG export skipped: {e})")
    print("Saved superimposed_with_timing_fix.html (+ .png if kaleido available)")
