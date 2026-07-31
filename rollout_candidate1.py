"""
Trained-model rollout + logging for candidate 1, using the minimal env.
Loads ppo_candidate1.zip + vecnormalize_candidate1.pkl, runs one
deterministic episode, logs action/torque/reward per step, saves CSV
and two diagnostic plots: (1) action per joint over time, (2) torque
per joint over time with gear-limit lines.

Usage:
    python rollout_candidate1.py
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from postural_env_minimal import Candidate1Env, JOINT_NAMES

MODEL_PATH = "ppo_candidate1"
VECNORM_PATH = "vecnormalize_candidate1.pkl"

def make_env():
    return TimeLimit(Candidate1Env(fixed_target=0.045), max_episode_steps=1000)

venv = DummyVecEnv([make_env])
venv = VecNormalize.load(VECNORM_PATH, venv)
venv.training = False
venv.norm_reward = False

model = PPO.load(MODEL_PATH)

raw_env = venv.venv.envs[0].unwrapped
gears = raw_env.model.actuator_gear[:, 0].copy()

obs = venv.reset()
rows = []
done = False
step = 0
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done_v, info = venv.step(action)
    done = bool(done_v[0])
    i0 = info[0]
    row = {"step": step, "reward": float(reward[0]),
           "com_y": float(i0["com_y"]), "target_y": float(i0["target_y"]),
           "h": float(i0["h"]), "failed": bool(i0["failed"]), "success": bool(i0["success"]),
           "xcom_y": float(i0["xcom_y"]), "com_y_dot": float(i0["com_y_dot"])}
    for j, name in enumerate(JOINT_NAMES):
        row[f"{name}_action"] = float(action[0][j])
        row[f"{name}_torque_nm"] = float(action[0][j]) * float(gears[j])
        row[f"{name}_angle_rad"] = float(raw_env.data.qpos[j])
    rows.append(row)
    step += 1

df = pd.DataFrame(rows)
os.makedirs("rollout_output", exist_ok=True)
df.to_csv("rollout_output/candidate1_rollout.csv", index=False)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df["step"], y=df["com_y"], mode="lines", name="com_y"))
fig2.add_hline(y=df["target_y"].iloc[0], line=dict(color="red", dash="dash"), name="target_y")
fig2.update_layout(title="CoM-y over time vs target", xaxis_title="step", yaxis_title="com_y (m)")
fig2.write_image("rollout_output/com_over_time.png")

last = df.iloc[-2]
print("\n--- Last step diagnostics ---")
print(f"xcom_y={last['xcom_y']:.4f}  base_half_width={raw_env.base_half_width:.4f}")
for name, low, high in zip(JOINT_NAMES, raw_env.fail_low, raw_env.fail_high):
    print(f"{name}: angle={last[f'{name}_angle_rad']:.4f}  limits=[{low:.4f}, {high:.4f}]")

print(f"Episode length: {len(df)} steps, target_y={df['target_y'].iloc[0]:.4f}")

print(f"Final com_y={df['com_y'].iloc[-1]:.4f}, final error={df['com_y'].iloc[-1]-df['target_y'].iloc[-1]:.4f}")
print(f"Failed: {df['failed'].iloc[-1]}, any success step: {df['success'].any()}")
for name in JOINT_NAMES:
    sat_frac = (df[f"{name}_action"].abs() > 0.95).mean()
    first_sat = df.index[df[f"{name}_action"].abs() > 0.95]
    first_sat_step = int(first_sat[0]) if len(first_sat) else -1
    print(f"{name}: saturated (|action|>0.95) {sat_frac*100:.1f}% of steps, first at step {first_sat_step}")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                     subplot_titles=("Action per joint over time", "Torque (Nm) per joint over time"))
colors = ["blue", "red", "green", "purple"]
for name, color in zip(JOINT_NAMES, colors):
    fig.add_trace(go.Scatter(x=df["step"], y=df[f"{name}_action"], mode="lines", name=f"{name} action", line=dict(color=color)), row=1, col=1)
for name, gear, color in zip(JOINT_NAMES, gears, colors):
    fig.add_trace(go.Scatter(x=df["step"], y=df[f"{name}_torque_nm"], mode="lines", name=f"{name} torque", line=dict(color=color), showlegend=False), row=2, col=1)
    fig.add_hline(y=gear, line=dict(color=color, dash="dot", width=1), row=2, col=1)
    fig.add_hline(y=-gear, line=dict(color=color, dash="dot", width=1), row=2, col=1)
fig.update_yaxes(title_text="Action [-1,1]", row=1, col=1)
fig.update_yaxes(title_text="Torque (Nm)", row=2, col=1)
fig.update_xaxes(title_text="Step", row=2, col=1)
fig.update_layout(title="Candidate 1 trained rollout: action vs torque saturation", height=700)
try:
    fig.write_image("rollout_output/action_torque_saturation.png")
except Exception as e:
    fig.write_html("rollout_output/action_torque_saturation.html")
    print(f"(PNG export failed, wrote HTML instead: {e})")

venv.close()
print("\nDone. See rollout_output/candidate1_rollout.csv and action_torque_saturation.png")
