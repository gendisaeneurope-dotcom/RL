"""Visualize / evaluate trained candidate policies for the AP task.
Supports candidate1 (F, no safety), candidate2 (XCoM), candidate3 (capture point).
Runs headless by default and saves plots + CSV for thesis reporting.

python watch.py candidate1 --episodes 20 --plot
python watch.py candidate2 --episodes 20 --plot
python watch.py candidate3 --episodes 20 --plot --render   # opens MuJoCo viewer too
"""
import argparse
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from candidate1_F_ap import Candidate1Env
from candidate2_ap import Candidate2Env
from candidate3_ap import Candidate3Env

CANDIDATES = {
    "candidate1": dict(env_cls=Candidate1Env, model="ppo_candidate1_F_ap",
                        vecnorm="vecnormalize_candidate1_F_ap.pkl", has_safety=False),
    "candidate2": dict(env_cls=Candidate2Env, model="ppo_candidate2_ap_failslope100",
                        vecnorm="vecnormalize_candidate2_ap_failslope100.pkl", has_safety=True),
    "candidate3": dict(env_cls=Candidate3Env, model="ppo_candidate3_ap",
                        vecnorm="vecnormalize_candidate3_ap.pkl", has_safety=True),
}


def save_fig(fig, path_noext):
    try:
        fig.write_image(f"{path_noext}.png")
    except Exception as e:
        fig.write_html(f"{path_noext}.html")
        print(f"  [PNG export failed ({type(e).__name__}), wrote {path_noext}.html instead]")


def make_plots(df, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{name}_watch_trajectories.csv"), index=False)

    summary = df.groupby("episode").agg(
        target_x=("target_x", "first"),
        final_com_x=("com_x", "last"),
        failed=("failed", "last"),
    ).reset_index()
    colors = ["red" if f else "royalblue" for f in summary["failed"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=summary["target_x"], y=summary["final_com_x"],
                              mode="markers", marker=dict(size=10, color=colors),
                              name="episodes"))
    lims = [summary["target_x"].min(), summary["target_x"].max()]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                              line=dict(dash="dot", color="gray"), name="Perfect tracking"))
    fig.update_layout(title=f"Final CoM-x vs target-x ({len(summary)} eval episodes, {name})",
                       xaxis_title="target_x (m)", yaxis_title="final CoM-x (m)")
    save_fig(fig, os.path.join(out_dir, f"{name}_target_tracking_scatter"))

    fig2 = go.Figure()
    for ep, d in df.groupby("episode"):
        fig2.add_trace(go.Scatter(x=d["step"], y=d["com_x"], mode="lines",
                                   opacity=0.6, showlegend=False))
    fig2.update_layout(title=f"CoM-x over time — {name}", xaxis_title="step", yaxis_title="CoM-x (m)")
    save_fig(fig2, os.path.join(out_dir, f"{name}_com_over_time"))

    print(f"Saved plots + CSV to {out_dir}/")
    print(summary.to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("candidate", choices=CANDIDATES.keys())
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--target", type=float, default=None,
                   help="Fixed target_x. Omit for random target each episode.")
    p.add_argument("--safety_weight", type=float, default=None,
                   help="Override safety_weight (candidate2/3 only). Omit to use script default.")
    p.add_argument("--render", action="store_true", help="Also open the MuJoCo viewer.")
    p.add_argument("--plot", action="store_true", help="Save plots + CSV.")
    p.add_argument("--out-dir", default=None, help="Defaults to ./eval_plots/<candidate>")
    a = p.parse_args()

    cfg = CANDIDATES[a.candidate]
    model = PPO.load(cfg["model"])

    env_kwargs = dict(fixed_target=a.target)
    if cfg["has_safety"] and a.safety_weight is not None:
        env_kwargs["safety_weight"] = a.safety_weight
    render_mode = "human" if a.render else None

    def make_env():
        e = cfg["env_cls"](render_mode=render_mode, **env_kwargs)
        return TimeLimit(e, max_episode_steps=1000)

    venv = DummyVecEnv([make_env])
    venv = VecNormalize.load(cfg["vecnorm"], venv)
    venv.training = False
    venv.norm_reward = False

    raw_env = venv.venv.envs[0].unwrapped
    rows = []

    for ep in range(a.episodes):
        obs = venv.reset()
        done = False
        step = 0
        info = {}
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = venv.step(act)
            done = done[0]
            info = infos[0]
            step += 1
            if a.plot:
                rows.append({"episode": ep + 1, "step": step,
                             "com_x": info.get("com_x"), "target_x": info.get("target_x"),
                             "failed": info.get("failed")})
        print(f"episode {ep+1}: target={info.get('target_x'):.4f} "
              f"final_com_x={info.get('com_x'):.4f} failed={info.get('failed')}")

    venv.close()

    if a.plot and rows:
        df = pd.DataFrame(rows)
        out_dir = a.out_dir or os.path.join("eval_plots", a.candidate)
        make_plots(df, out_dir, a.candidate)


if __name__ == "__main__":
    main()