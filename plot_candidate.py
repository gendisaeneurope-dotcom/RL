"""
Adapter to reuse plot_results.py's plotting logic against the new
standalone candidate1_target.py / candidate2_xcom.py outputs, which
don't have postural_env.py's run_dir/config.json/model structure.

Usage:
    python plot_candidate.py candidate1
    python plot_candidate.py candidate2_xcom
"""
import os
import sys
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import load_results

JOINT_NAMES = ["ankle_eversion", "ankle_flexion", "hip_abduction", "hip_flexion"]

CONFIGS = {
    "candidate1_A": dict(module="candidate1_A", cls="Candidate1Env",
                        model="ppo_candidate1_A", vecnorm="vecnormalize_candidate1_A.pkl",
                        log_dir="./training_logs_candidate1_A/"),
    "candidate2_xcom": dict(module="candidate2_xcom", cls="Candidate2Env",
                             model="ppo_candidate2_xcom", vecnorm="vecnormalize_candidate2_xcom.pkl",
                             log_dir="./training_logs_candidate2_xcom/"),
    "candidate1_B": dict(module="candidate1_B", cls="Candidate1Env",
                          model="ppo_candidate1_B", vecnorm="vecnormalize_candidate1_B.pkl",
                          log_dir="./training_logs_candidate1_B/"),
    "candidate1_C": dict(module="candidate1_C", cls="Candidate1Env",
                          model="ppo_candidate1_C", vecnorm="vecnormalize_candidate1_C.pkl",
                          log_dir="./training_logs_candidate1_C/"),
    "candidate1_D": dict(module="candidate1_D", cls="Candidate1Env",
                          model="ppo_candidate1_D", vecnorm="vecnormalize_candidate1_D.pkl",
                          log_dir="./training_logs_candidate1_D/"),
    "candidate1_E": dict(module="candidate1_E", cls="Candidate1Env",
                              model="ppo_candidate1_E", vecnorm="vecnormalize_candidate1_E.pkl",
                              log_dir="./training_logs_candidate1_E/"),
    "candidate1_F": dict(module="candidate1_F", cls="Candidate1Env",
                                  model="ppo_candidate1_F", vecnorm="vecnormalize_candidate1_F.pkl",
                                  log_dir="./training_logs_candidate1_F/"),
}


def _save(fig, path_no_ext):
    try:
        fig.write_image(path_no_ext + ".png")
    except Exception as e:
        fig.write_html(path_no_ext + ".html")
        print(f" (PNG export failed [{type(e).__name__}], wrote {path_no_ext}.html instead)")


def collect_episodes(cfg, EnvClass, n_episodes=20):
    model = PPO.load(cfg["model"])
    rows = []
    for ep in range(1, n_episodes + 1):
        venv = DummyVecEnv([lambda: TimeLimit(EnvClass(), max_episode_steps=1000)])
        venv = VecNormalize.load(cfg["vecnorm"], venv)
        venv.training = False
        venv.norm_reward = False

        obs = venv.reset()
        raw = venv.venv.envs[0].unwrapped
        gears = raw.model.actuator_gear[:, 0].copy()
        total_reward, step_idx = 0.0, 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            angles = raw.data.qpos[:4].copy()

            row = {"episode": ep, "step": step_idx + 1}
            for i, name in enumerate(JOINT_NAMES):
                row[f"{name}_deg"] = float(angles[i]) * 180 / math.pi
                row[f"{name}_torque_nm"] = float(action[0][i]) * float(gears[i])

            obs, reward, done_v, info = venv.step(action)
            done = bool(done_v[0])
            total_reward += float(reward[0])
            step_idx += 1

            i0 = info[0]
            row.update({
                "reward": float(reward[0]), "cum_reward": total_reward,
                "com_y": float(i0["com_y"]), "target_y": float(i0["target_y"]),
                "h": float(i0["h"]), "xcom_y": float(i0["xcom_y"]),
                "failed": bool(i0["failed"]),
            })
            rows.append(row)
        venv.close()

    df = pd.DataFrame(rows)
    summary = df.groupby("episode").agg(
        total_reward=("reward", "sum"), steps=("step", "max"),
        target_y=("target_y", "first"), final_com_y=("com_y", "last"),
        final_h=("h", "last"), failed=("failed", "last"),
    ).reset_index()
    summary["final_error"] = summary["final_com_y"] - summary["target_y"]
    return df, summary


def make_plots(out_dir, df, summary, base_half_width, joint_gears, title_tag, log_dir):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "trajectories.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "episode_summary.csv"), index=False)
    print(summary)

    d = df[df["episode"] == 1]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=("Joint angle trajectories (episode 1)",
                                         "CoM-y vs. target-y (episode 1)"))
    for name in JOINT_NAMES:
        fig.add_trace(go.Scatter(x=d["step"], y=d[f"{name}_deg"], mode="lines", name=f"{name} angle"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["step"], y=d["com_y"], mode="lines", name="CoM y", line=dict(color="royalblue")), row=2, col=1)
    fig.add_trace(go.Scatter(x=d["step"], y=d["target_y"], mode="lines", name="Target y", line=dict(color="firebrick", dash="dash")), row=2, col=1)
    fig.update_layout(title=f"Joint angles vs. CoM position (episode 1, {title_tag})", height=700)
    _save(fig, os.path.join(out_dir, "joint_angles_vs_com"))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["step"], y=d["xcom_y"], mode="lines", name="XCoM y", line=dict(color="purple")))
    fig.add_trace(go.Scatter(x=d["step"], y=d["com_y"], mode="lines", name="CoM y", line=dict(color="royalblue", dash="dot")))
    fig.add_hline(y=base_half_width, line=dict(color="red", dash="dash"), annotation_text="base-of-support boundary")
    fig.add_hline(y=-base_half_width, line=dict(color="red", dash="dash"))
    fig.update_layout(title=f"XCoM-y vs. base-of-support boundary (episode 1, {title_tag})")
    _save(fig, os.path.join(out_dir, "xcom_vs_boundary"))

    colors = ["blue", "red", "green", "purple"]
    fig = go.Figure()
    for name, color in zip(JOINT_NAMES, colors):
        fig.add_trace(go.Scatter(x=d["step"], y=d[f"{name}_torque_nm"], mode="lines", name=f"{name} torque", line=dict(color=color)))
    for name, gear, color in zip(JOINT_NAMES, joint_gears, colors):
        fig.add_hline(y=gear, line=dict(color=color, dash="dot", width=1))
        fig.add_hline(y=-gear, line=dict(color=color, dash="dot", width=1))
    fig.update_layout(title=f"Joint torque trajectories (episode 1, {title_tag})")
    _save(fig, os.path.join(out_dir, "joint_torque_trajectories"))

    fig = go.Figure()
    colors = ["red" if f else "royalblue" for f in summary["failed"]]
    fig.add_trace(go.Scatter(x=summary["target_y"], y=summary["final_com_y"], mode="markers", marker=dict(size=9, color=colors)))
    lims = [summary["target_y"].min(), summary["target_y"].max()]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="Perfect tracking", line=dict(color="gray", dash="dot")))
    fig.update_layout(title=f"Final CoM-y vs. target-y ({len(summary)} eval episodes, {title_tag})")
    _save(fig, os.path.join(out_dir, "target_tracking_scatter"))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=summary["episode"], y=summary["total_reward"], mode="lines+markers"))
    fig.update_layout(title=f"Episode reward ({len(summary)} eval eps, {title_tag})")
    _save(fig, os.path.join(out_dir, "episode_rewards"))

    try:
        df_train = load_results(log_dir)
    except Exception as e:
        print(f"(no training monitor log found in {log_dir}: {e})")
        return
    df_train["episode"] = range(len(df_train))
    df_train["r_smoothed"] = df_train["r"].rolling(20, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_train["episode"], y=df_train["r"], mode="lines", name="Raw reward", opacity=0.3))
    fig.add_trace(go.Scatter(x=df_train["episode"], y=df_train["r_smoothed"], mode="lines", name="Smoothed (20-ep)"))
    fig.update_layout(title=f"Training reward per episode ({title_tag})")
    _save(fig, os.path.join(out_dir, "training_reward"))


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "candidate1"
    cfg = CONFIGS[key]
    module = __import__(cfg["module"])
    EnvClass = getattr(module, cfg["cls"])

    df, summary = collect_episodes(cfg, EnvClass)
    probe = EnvClass()
    bhw = probe.base_half_width
    gears = probe.model.actuator_gear[:, 0].copy()

    out_dir = os.path.join(cfg["log_dir"], "plots")
    make_plots(out_dir, df, summary, bhw, gears, key, cfg["log_dir"])
    print(f"\nPlots saved to {out_dir}/")
