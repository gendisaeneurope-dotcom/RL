"""Plot the sim policy's trajectory against ONE real subject's trial,
aligned in time -- not just the final position (that's compare_to_real.py).

Uses real_trials_extracted.csv (per-timestep, not the summary CSV).

  python compare_to_real_plot.py runs/none_w0_s0 real_trials_extracted.csv
"""
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from postural_env import PosturalEnv

run_dir = sys.argv[1]
extracted_path = sys.argv[2]

df = pd.read_csv(extracted_path)

model = PPO.load(f"{run_dir}/model")

fig = go.Figure()

# same sign convention fixed in compare_to_real.py: right = +, left = -
for side, sign in [("right", +1), ("left", -1)]:
    sub = df[df.side == side]
    # pick the trial_group closest to the MEDIAN length for this side --
    # picking the longest trial grabbed outliers (one was 77s vs a ~4.5s
    # median), not representative trials
    sizes = sub.groupby("trial_group").size()
    median_n = sizes.median()
    best_tg = (sizes - median_n).abs().idxmin()
    trial = sub[sub.trial_group == best_tg].sort_values("t_rel")
    t_real = trial["t_rel"].values - trial["t_rel"].values[0]
    com_real_mm = trial["com_m"].values * 1000
    target = sign * abs(com_real_mm[-1]) / 1000  # match this trial's own endpoint as target

    def make_env(t=target):
        e = PosturalEnv(safety="none", safety_weight=0.0, fixed_target=t)
        e = TimeLimit(e, max_episode_steps=1000)
        e.reset(seed=0)
        return e
    venv = DummyVecEnv([make_env])
    venv = VecNormalize.load(f"{run_dir}/vecnormalize.pkl", venv)
    venv.training = False
    venv.norm_reward = False
    obs = venv.reset()

    sim_com_mm, sim_t = [], []
    step_dt = venv.venv.envs[0].unwrapped.step_dt
    for i in range(1000):
        act, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = venv.step(act)
        sim_com_mm.append(info[0]["com_y"] * 1000)
        sim_t.append((i + 1) * step_dt)
        if done[0]:
            break
    venv.close()

    fig.add_trace(go.Scatter(x=t_real, y=com_real_mm, mode="lines",
                             name=f"real subject ({side}, trial {best_tg})",
                             line=dict(dash="solid")))
    fig.add_trace(go.Scatter(x=sim_t, y=sim_com_mm, mode="lines",
                             name=f"sim policy ({side}, target={target*1000:.1f}mm)",
                             line=dict(dash="dot")))

    print(f"{side}: real trial duration={t_real[-1]:.2f}s ({len(t_real)} samples), "
          f"sim ran {sim_t[-1]:.2f}s ({len(sim_t)} steps), "
          f"real final={com_real_mm[-1]:.1f}mm, sim final={sim_com_mm[-1]:.1f}mm")

fig.update_layout(title="Sim policy vs. real subject trajectory (CoM-y over time)",
                  xaxis_title="Time (s)", yaxis_title="CoM y (mm)")
try:
    fig.write_image("compare_to_real_trajectory.png")
    print("\nSaved: compare_to_real_trajectory.png")
except Exception as e:
    fig.write_html("compare_to_real_trajectory.html")
    print(f"\n(PNG export failed [{type(e).__name__}], saved "
          f"compare_to_real_trajectory.html instead -- open in a browser)")
