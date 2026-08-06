"""
Adapter to reuse plot_results.py's plotting logic against the new
standalone candidate1_target.py / candidate2_xcom.py outputs, which
don't have postural_env.py's run_dir/config.json/model structure.

Usage:
    python plot_candidate_ap.py candidate1_F_ap
    python plot_candidate_ap.py candidate2_ap
    python plot_candidate_ap.py candidate3_ap
    python plot_candidate_ap.py candidate2_ap --meanstd
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
    "candidate1_F_ap": dict(module="candidate1_F_ap", cls="Candidate1Env",
                             model="ppo_candidate1_F_ap", vecnorm="vecnormalize_candidate1_F_ap.pkl",
                             log_dir="./training_logs_candidate1_F_ap/"),
    "candidate1_ap_start": dict(module="candidate1_F_ap", cls="Candidate1Env",
                                 model="ppo_candidate1_ap_start", vecnorm="vecnormalize_candidate1_ap_start.pkl",
                                 log_dir="./training_logs_candidate1_ap_start/"),
    "candidate2_ap": dict(module="candidate2_ap", cls="Candidate2Env",
                           model="ppo_candidate2_ap", vecnorm="vecnormalize_candidate2_ap.pkl",
                           log_dir="./training_logs_candidate2_ap/"),
    "candidate2_ap_start": dict(module="candidate2_ap", cls="Candidate2Env",
                                 model="ppo_candidate2_ap_start", vecnorm="vecnormalize_candidate2_ap_start.pkl",
                                 log_dir="./training_logs_candidate2_ap_start/"),
    "candidate2_ap_sw005_01disturb": dict(module="candidate2_ap_disturb", cls="Candidate2Env",
                                            model="ppo_candidate2_ap_sw005_01disturb", vecnorm="vecnormalize_candidate2_ap_sw005_01disturb.pkl",
                                            log_dir="./training_logs_candidate2_ap_sw005_01disturb/"),
    "candidate3_ap": dict(module="candidate3_ap", cls="Candidate3Env",
                           model="ppo_candidate3_ap", vecnorm="vecnormalize_candidate3_ap.pkl",
                           log_dir="./training_logs_candidate3_ap/"),
    "candidate3_ap_start": dict(module="candidate3_ap", cls="Candidate3Env",
                                 model="ppo_candidate3_ap_start", vecnorm="vecnormalize_candidate3_ap_start.pkl",
                                 log_dir="./training_logs_candidate3_ap_start/"),
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

        venv.seed(ep)
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
                "com_x": float(i0["com_x"]), "com_y": float(i0.get("com_y", 0.0)),
                "target_x": float(i0["target_x"]),
                "h": float(i0["h"]), "xcom_x": float(i0["xcom_x"]),
                "failed": bool(i0["failed"]),
            })
            rows.append(row)
        venv.close()

    df = pd.DataFrame(rows)
    summary = df.groupby("episode").agg(
        total_reward=("reward", "sum"), steps=("step", "max"),
        target_x=("target_x", "first"), final_com_x=("com_x", "last"),
        final_h=("h", "last"), failed=("failed", "last"),
    ).reset_index()
    summary["final_error"] = summary["final_com_x"] - summary["target_x"]
    return df, summary


def collect_meanstd(cfg, EnvClass, n_episodes=20, deterministic=False):
    """
    Collects per-episode CoM-x TRACKING ERROR trajectories (com_x - target_x),
    not raw CoM-x. Each episode samples its own random in-distribution target
    (same default sampling the env normally uses), matching collect_episodes.
    Plotting the error instead of raw position means episodes with different
    targets are directly comparable, and there is no fixed_target to hardcode
    or let go stale when the training target range changes.
    """
    model = PPO.load(cfg["model"])
    all_err, targets_used = [], []
    for ep in range(n_episodes):
        venv = DummyVecEnv([lambda: TimeLimit(EnvClass(), max_episode_steps=1000)])
        venv = VecNormalize.load(cfg["vecnorm"], venv)
        venv.training = False
        venv.norm_reward = False

        venv.seed(ep)
        obs = venv.reset()

        done, err_traj, tgt = False, [], None
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, done_v, info = venv.step(action)
            done = bool(done_v[0])
            i0 = info[0]
            if tgt is None:
                tgt = float(i0["target_x"])
            err_traj.append(float(i0["com_x"]) - float(i0["target_x"]))
        venv.close()
        all_err.append(err_traj)
        targets_used.append(tgt)

    max_len = max(len(t) for t in all_err)
    padded = np.array([t + [t[-1]] * (max_len - len(t)) for t in all_err])
    return padded.mean(axis=0), padded.std(axis=0), targets_used


def meanstd_plot(cfg, EnvClass, out_dir, title_tag, n_episodes=20, deterministic=False):
    mean_err, std_err, targets_used = collect_meanstd(
        cfg, EnvClass, n_episodes=n_episodes, deterministic=deterministic
    )
    x = np.arange(len(mean_err))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=mean_err, mode="lines", name="CoM-x error mean", line=dict(color="royalblue")))
    fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]),
                              y=np.concatenate([mean_err + std_err, (mean_err - std_err)[::-1]]),
                              fill="toself", fillcolor="royalblue", opacity=0.2,
                              line=dict(width=0), showlegend=False))
    fig.add_hline(y=0.0, line=dict(dash="dot", color="firebrick"), annotation_text="perfect tracking")
    tmin, tmax = min(targets_used), max(targets_used)
    fig.update_layout(
        title=f"CoM-x tracking error mean +/- std over time "
              f"(targets sampled in [{tmin:.3f}, {tmax:.3f}], {title_tag})",
        xaxis_title="step", yaxis_title="com_x - target_x (m)")
    _save(fig, os.path.join(out_dir, "com_x_meanstd"))


def make_plots(out_dir, df, summary, base_half_length, joint_gears, title_tag, log_dir):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "trajectories.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "episode_summary.csv"), index=False)
    print(summary)

    d = df[df["episode"] == 1]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=("Joint angle trajectories (episode 1)",
                                         "CoM-x vs. target-x (episode 1)"))
    for name in JOINT_NAMES:
        fig.add_trace(go.Scatter(x=d["step"], y=d[f"{name}_deg"], mode="lines", name=f"{name} angle"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["step"], y=d["com_x"], mode="lines", name="CoM x", line=dict(color="royalblue")), row=2, col=1)
    fig.add_trace(go.Scatter(x=d["step"], y=d["target_x"], mode="lines", name="Target x", line=dict(color="firebrick", dash="dash")), row=2, col=1)
    fig.update_layout(title=f"Joint angles vs. CoM position (episode 1, {title_tag})", height=700)
    _save(fig, os.path.join(out_dir, "joint_angles_vs_com"))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["step"], y=d["xcom_x"], mode="lines", name="XCoM x", line=dict(color="purple")))
    fig.add_trace(go.Scatter(x=d["step"], y=d["com_x"], mode="lines", name="CoM x", line=dict(color="royalblue", dash="dot")))
    fig.add_hline(y=base_half_length, line=dict(color="red", dash="dash"), annotation_text="base-of-support boundary")
    fig.add_hline(y=-base_half_length, line=dict(color="red", dash="dash"))
    fig.update_layout(title=f"XCoM-x vs. base-of-support boundary (episode 1, {title_tag})")
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
    fig.add_trace(go.Scatter(x=summary["target_x"], y=summary["final_com_x"], mode="markers", marker=dict(size=9, color=colors)))
    lims = [summary["target_x"].min(), summary["target_x"].max()]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="Perfect tracking", line=dict(color="gray", dash="dot")))
    fig.update_layout(title=f"Final CoM-x vs. target-x ({len(summary)} eval episodes, {title_tag})")
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


def com_xy_plot(df, outdir, title_tag):
    """
    CoM-x vs. CoM-y, NOT vs. time -- shows the actual 2D movement path of
    the center of mass across the base of support. One line per episode.
    """
    fig = go.Figure()
    for ep, d in df.groupby("episode"):
        t = d["target_x"].iloc[0]
        fig.add_trace(go.Scatter(
            x=d["com_x"], y=d["com_y"], mode="lines",
            name=f"ep {ep} (target={t:.3f})", opacity=0.6
        ))
    fig.update_layout(title=f"CoM-x vs. CoM-y trajectory, all episodes {title_tag}",
                       xaxis_title="CoM x (m, anterior-posterior)",
                       yaxis_title="CoM y (m, mediolateral)")
    _save(fig, os.path.join(outdir, "com_x_vs_y"))


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "candidate1"
    do_meanstd = "--meanstd" in sys.argv
    cfg = CONFIGS[key]
    module = __import__(cfg["module"])
    EnvClass = getattr(module, cfg["cls"])

    df, summary = collect_episodes(cfg, EnvClass)
    probe = EnvClass()
    bhl = probe.base_half_length
    gears = probe.model.actuator_gear[:, 0].copy()

    out_dir = os.path.join(cfg["log_dir"], "plots")
    make_plots(out_dir, df, summary, bhl, gears, key, cfg["log_dir"])
    com_xy_plot(df, out_dir, key)

    if do_meanstd:
        meanstd_plot(cfg, EnvClass, out_dir, key)

    print(f"\nPlots saved to {out_dir}/")
