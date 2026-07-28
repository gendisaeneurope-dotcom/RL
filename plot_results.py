"""Plots for any trained run. Same style as your original per-candidate plot
scripts, but one file instead of three.

  python plot_results.py runs/none_w0_s0
  python plot_results.py runs/xcom_w1_s0
  python plot_results.py runs/joint_w1_s0

Output goes to runs/<name>/plots/
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

from postural_env import PosturalEnv, JOINT_NAMES, load_run_config


def collect_episodes(run_dir, n_episodes=20):
    """Roll out the trained policy and record every step. Returns a
    per-step dataframe and a per-episode summary dataframe."""
    cfg = load_run_config(run_dir)
    model = PPO.load(f"{run_dir}/model")

    rows = []
    for ep in range(1, n_episodes + 1):
        venv = DummyVecEnv([lambda: TimeLimit(
            PosturalEnv(mode=cfg["mode"], safety=cfg["safety"], safety_weight=0.0,
                       use_shaping=cfg.get("use_shaping", False)),
            max_episode_steps=1000)])
        venv = VecNormalize.load(f"{run_dir}/vecnormalize.pkl", venv)
        venv.training = False
        venv.norm_reward = False

        obs = venv.reset()
        raw = venv.venv.envs[0].unwrapped
        total_reward, step_idx, info = 0.0, 0, [{}]
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            angles = raw.data.qpos[:4].copy()

            row = {"episode": ep, "step": step_idx + 1}
            for i, name in enumerate(JOINT_NAMES):
                row[f"{name}_deg"] = float(angles[i]) * 180 / math.pi
                row[f"{name}_action"] = float(action[0][i])

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

    agg_dict = {
        "total_reward": ("reward", "sum"), "steps": ("step", "max"),
        "target_y": ("target_y", "first"),
        "final_com_y": ("com_y", "last"),
        "final_h": ("h", "last"),
        "max_abs_xcom_y": ("xcom_y", lambda s: np.max(np.abs(s))),
        "failed": ("failed", "last"),
    }
    for name in JOINT_NAMES:
        agg_dict[f"mean_abs_{name}_deg"] = (f"{name}_deg", lambda s: np.mean(np.abs(s)))
        agg_dict[f"max_abs_{name}_deg"] = (f"{name}_deg", lambda s: np.max(np.abs(s)))
    summary = df.groupby("episode").agg(**agg_dict).reset_index()
    summary["final_error"] = summary["final_com_y"] - summary["target_y"]
    return df, summary


def _save(fig, path_no_ext):
    """Write PNG via kaleido; fall back to HTML if Chrome isn't available.
    An interactive HTML file opens in any browser and is fine for a thesis
    draft -- swap to PNG once Chrome is present for the final figures."""
    try:
        fig.write_image(path_no_ext + ".png")
    except Exception as e:
        fig.write_html(path_no_ext + ".html")
        print(f"  (PNG export failed [{type(e).__name__}], wrote "
              f"{path_no_ext}.html instead -- open it in a browser. "
              f"Run 'plotly_get_chrome' once to enable PNG export.)")


def make_plots(run_dir, df, summary, base_half_width, title_tag):
    out_dir = os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "trajectories.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "episode_summary.csv"), index=False)
    print(summary)

    d = df[df["episode"] == 1]

    # 1. joint angles + CoM-y vs target-y, episode 1
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Joint angle trajectories (episode 1)",
                                        "CoM-y vs. target-y (episode 1)"))
    for name in JOINT_NAMES:
        fig.add_trace(go.Scatter(x=d["step"], y=d[f"{name}_deg"], mode="lines", name=f"{name} angle"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["step"], y=d["com_y"], mode="lines", name="CoM y",
                              line=dict(color="royalblue")), row=2, col=1)
    fig.add_trace(go.Scatter(x=d["step"], y=d["target_y"], mode="lines", name="Target y",
                              line=dict(color="firebrick", dash="dash")), row=2, col=1)
    fig.update_layout(title=f"Joint angles vs. CoM position (episode 1, {title_tag})", height=700)
    fig.update_yaxes(title_text="Angle (deg)", row=1, col=1)
    fig.update_yaxes(title_text="y (m)", row=2, col=1)
    fig.update_xaxes(title_text="Step", row=2, col=1)
    _save(fig, os.path.join(out_dir, "joint_angles_vs_com"))

    # 2. XCoM vs base-of-support boundary, episode 1 (meaningful for every
    # candidate -- shows whether the OTHER candidates' policies happen to
    # respect the same stability boundary candidate 2 was explicitly trained on)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["step"], y=d["xcom_y"], mode="lines", name="XCoM y", line=dict(color="purple")))
    fig.add_trace(go.Scatter(x=d["step"], y=d["com_y"], mode="lines", name="CoM y", line=dict(color="royalblue", dash="dot")))
    fig.add_hline(y=base_half_width, line=dict(color="red", dash="dash"), annotation_text="base-of-support boundary")
    fig.add_hline(y=-base_half_width, line=dict(color="red", dash="dash"))
    fig.update_layout(title=f"XCoM-y vs. base-of-support boundary (episode 1, {title_tag})")
    fig.update_xaxes(title_text="Step")
    fig.update_yaxes(title_text="y (m)")
    _save(fig, os.path.join(out_dir, "xcom_vs_boundary"))

    # 3. joint actions, episode 1
    fig = go.Figure()
    for name in JOINT_NAMES:
        fig.add_trace(go.Scatter(x=d["step"], y=d[f"{name}_action"], mode="lines", name=f"{name} action"))
    fig.update_layout(title=f"Joint action trajectories (episode 1, {title_tag})")
    fig.update_xaxes(title_text="Step")
    fig.update_yaxes(title_text="Action (normalized, -1 to 1)")
    _save(fig, os.path.join(out_dir, "joint_action_trajectories"))

    # 4. final com_y vs target_y scatter, all episodes
    fig = go.Figure()
    colors = ["red" if f else "royalblue" for f in summary["failed"]]
    fig.add_trace(go.Scatter(x=summary["target_y"], y=summary["final_com_y"], mode="markers",
                              name="Episodes", marker=dict(size=9, color=colors)))
    lims = [summary["target_y"].min(), summary["target_y"].max()]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="Perfect tracking (y=x)",
                              line=dict(color="gray", dash="dot")))
    fig.update_layout(title=f"Final CoM-y vs. target-y ({len(summary)} eval episodes, {title_tag})<br>"
                            f"<sub>red = episode failed before completing</sub>")
    fig.update_xaxes(title_text="Target y (m)")
    fig.update_yaxes(title_text="Final CoM y (m)")
    _save(fig, os.path.join(out_dir, "target_tracking_scatter"))

    # 5. episode reward summary
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=summary["episode"], y=summary["total_reward"], mode="lines+markers", name="Reward"))
    fig.update_layout(title=f"Episode reward ({len(summary)} eval eps, {title_tag})")
    fig.update_xaxes(title_text="Episode")
    fig.update_yaxes(title_text="Reward")
    _save(fig, os.path.join(out_dir, "episode_rewards"))

    # 6. training curves, if the training log exists
    log_dir = run_dir  # train.py writes progress.csv/monitor.csv into run_dir
    try:
        df_train = load_results(log_dir)
    except Exception as e:
        print(f"(no training monitor log found in {log_dir}: {e})")
        return
    df_train["episode"] = range(len(df_train))
    df_train["r_smoothed"] = df_train["r"].rolling(20, min_periods=1).mean()
    df_train["l_smoothed"] = df_train["l"].rolling(20, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_train["episode"], y=df_train["r"], mode="lines", name="Raw reward", opacity=0.3))
    fig.add_trace(go.Scatter(x=df_train["episode"], y=df_train["r_smoothed"], mode="lines", name="Smoothed (20-ep)"))
    fig.update_layout(title=f"Training reward per episode ({title_tag})")
    fig.update_xaxes(title_text="Episode (during training)")
    fig.update_yaxes(title_text="Reward")
    _save(fig, os.path.join(out_dir, "training_reward"))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_train["episode"], y=df_train["l"], mode="lines", name="Episode length (raw)", opacity=0.3))
    fig.add_trace(go.Scatter(x=df_train["episode"], y=df_train["l_smoothed"], mode="lines", name="Smoothed (20-ep)"))
    fig.update_layout(title=f"Episode length during training ({title_tag})")
    fig.update_xaxes(title_text="Episode (during training)")
    fig.update_yaxes(title_text="Steps per episode")
    _save(fig, os.path.join(out_dir, "training_episode_length"))

    # reward PER STEP: total reward divided by episode length. Total reward
    # alone scales with how long the episode survived, so it can look like
    # it's rising just because episodes got longer, not because per-step
    # behavior improved. This is the fairer curve for judging the policy.
    df_train["reward_per_step"] = df_train["r"] / df_train["l"]
    df_train["reward_per_step_smoothed"] = df_train["reward_per_step"].rolling(20, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_train["episode"], y=df_train["reward_per_step"], mode="lines", name="Reward/step (raw)", opacity=0.3))
    fig.add_trace(go.Scatter(x=df_train["episode"], y=df_train["reward_per_step_smoothed"], mode="lines", name="Smoothed (20-ep)"))
    fig.update_layout(title=f"Mean reward per step during training ({title_tag})")
    fig.update_xaxes(title_text="Episode (during training)")
    fig.update_yaxes(title_text="Reward per step")
    _save(fig, os.path.join(out_dir, "training_reward_per_step"))


def com_over_time_plot(df, out_dir, title_tag):
    """CoM-y vs time, all episodes overlaid, each labeled by its target.
    Matches the standing-still-style plot: one line per episode, so drift/
    settling behavior is visible at a glance across the whole eval set."""
    fig = go.Figure()
    for ep, d in df.groupby("episode"):
        t = d["target_y"].iloc[0]
        fig.add_trace(go.Scatter(x=d["step"], y=d["com_y"], mode="lines",
                                 name=f"ep{ep} (target={t:.3f})", opacity=0.6))
    fig.update_layout(title=f"CoM-y over time, all episodes ({title_tag})")
    fig.update_xaxes(title_text="Step")
    fig.update_yaxes(title_text="CoM y (m)")
    _save(fig, os.path.join(out_dir, "com_over_time_all_episodes"))


if __name__ == "__main__":
    run_dir = sys.argv[1].rstrip("/")
    title_tag = os.path.basename(run_dir)
    df, summary = collect_episodes(run_dir)
    # base_half_width is a model constant, same for every candidate
    from postural_env import PosturalEnv as _E
    bhw = _E(safety="none", safety_weight=0.0).base_half_width
    make_plots(run_dir, df, summary, bhw, title_tag)
    com_over_time_plot(df, os.path.join(run_dir, "plots"), title_tag)
    print(f"\nPlots saved to {run_dir}/plots/")