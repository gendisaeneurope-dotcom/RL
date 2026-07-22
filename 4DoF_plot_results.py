import os
output_dir = "./output_4dof/"
os.makedirs(output_dir, exist_ok=True)

import math
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from RL_4DoF import My4DOFEnv, JOINT_NAMES, N_JOINTS
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import load_results

# ── Load trained model ──
model = PPO.load("ppo_ankle_knee_hip_trunk")  #Rename the model file to match your saved model name

# ── Roll out 20 evaluation episodes ──
env = TimeLimit(My4DOFEnv(), max_episode_steps=1000)

rows = []
for ep in range(1, 21):
    obs, info = env.reset()
    total_reward = 0.0
    done = False
    step_idx = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        row = {"episode": ep, "step": step_idx + 1}
        for i, name in enumerate(JOINT_NAMES):
            row[f"{name}_deg"] = float(obs[i]) * 180 / math.pi
            row[f"{name}_action"] = float(action[i])

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        step_idx += 1

        row.update({
            "reward": float(reward), "cum_reward": total_reward,
            "terminated": terminated, "truncated": truncated
        })
        rows.append(row)

        done = terminated or truncated

env.close()

df = pd.DataFrame(rows)
df.to_csv(os.path.join(output_dir, "4dof_trunk_trajectories.csv"), index=False)

agg_dict = {"total_reward": ("reward", "sum"), "steps": ("step", "max")}
for name in JOINT_NAMES:
    agg_dict[f"mean_abs_{name}_deg"] = (f"{name}_deg", lambda s: np.mean(np.abs(s)))
    agg_dict[f"max_abs_{name}_deg"] = (f"{name}_deg", lambda s: np.max(np.abs(s)))
summary = df.groupby("episode").agg(**agg_dict).reset_index()

summary.to_csv(os.path.join(output_dir, "4dof_trunk_episode_summary.csv"), index=False)
print(summary)


fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=summary["episode"], y=summary["total_reward"], mode="lines+markers", name="Reward"))
fig1.update_layout(title={"text": "Episode reward (20 eval eps, 4-DOF)<br><span style='font-size: 18px; font-weight: normal;'>Ankle-knee-hip-trunk model under energy reward</span>"})
fig1.update_xaxes(title_text="Episode")
fig1.update_yaxes(title_text="Reward")
fig1.write_image(os.path.join(output_dir, "episode_rewards_4dof.png"))
with open(os.path.join(output_dir, "episode_rewards_4dof.png.meta.json"), "w") as f:
    json.dump({"caption": "Episode rewards over 20 evaluation episodes (4-DOF ankle-knee-hip-trunk)",
               "description": "Line chart of total reward for each evaluation episode."}, f)


d = df[df["episode"] == 1]
fig2 = go.Figure()
for name in JOINT_NAMES:
    fig2.add_trace(go.Scatter(x=d["step"], y=d[f"{name}_deg"], mode="lines", name=f"{name.capitalize()} angle"))
fig2.update_layout(title={"text": "Joint angle trajectory (episode 1, 4-DOF)<br><span style='font-size: 18px; font-weight: normal;'>Recovery-then-hold behavior across all 4 joints</span>"})
fig2.update_xaxes(title_text="Step")
fig2.update_yaxes(title_text="Angle (deg)")
fig2.write_image(os.path.join(output_dir, "joint_angle_trajectories_4dof.png"))
with open(os.path.join(output_dir, "joint_angle_trajectories_4dof.png.meta.json"), "w") as f:
    json.dump({"caption": "Ankle, knee, hip, and trunk angle trajectories for episode 1",
               "description": "Line chart showing all 4 joint angles across timesteps."}, f)


df_train = load_results("./training_logs_4dof/")
df_train["episode"] = range(len(df_train))
df_train["r_smoothed"] = df_train["r"].rolling(20, min_periods=1).mean()

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df_train["episode"], y=df_train["r"], mode="lines", name="Raw reward", opacity=0.3))
fig3.add_trace(go.Scatter(x=df_train["episode"], y=df_train["r_smoothed"], mode="lines", name="Smoothed (20-ep rolling mean)"))
fig3.update_layout(title={"text": "Training reward per episode (4-DOF)<br><span style='font-size: 18px; font-weight: normal;'>Reward should trend toward the success bonus as policy learns</span>"})
fig3.update_xaxes(title_text="Episode (during training)")
fig3.update_yaxes(title_text="Reward")
fig3.write_image(os.path.join(output_dir, "training_reward_4dof.png"))
with open(os.path.join(output_dir, "training_reward_4dof.png.meta.json"), "w") as f:
    json.dump({"caption": "Reward per episode during 4-DOF PPO training",
               "description": "Line chart showing raw and smoothed reward across training episodes."}, f)


df_train["l_smoothed"] = df_train["l"].rolling(20, min_periods=1).mean()
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=df_train["episode"], y=df_train["l"], mode="lines", name="Episode length (raw)", opacity=0.3))
fig4.add_trace(go.Scatter(x=df_train["episode"], y=df_train["l_smoothed"], mode="lines", name="Episode length (smoothed)"))
fig4.update_layout(title={"text": "Episode length during training (4-DOF)<br><span style='font-size: 18px; font-weight: normal;'>Longer episodes reflect improved balance duration</span>"})
fig4.update_xaxes(title_text="Episode (during training)")
fig4.update_yaxes(title_text="Steps per episode")
fig4.write_image(os.path.join(output_dir, "training_episode_length_4dof.png"))
with open(os.path.join(output_dir, "training_episode_length_4dof.png.meta.json"), "w") as f:
    json.dump({"caption": "Episode length per episode during 4-DOF PPO training",
               "description": "Line chart of raw and smoothed episode length across training episodes."}, f)


df_train["reward_per_step"] = df_train["r"] / df_train["l"]
df_train["reward_per_step_smoothed"] = df_train["reward_per_step"].rolling(20, min_periods=1).mean()
fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=df_train["episode"], y=df_train["reward_per_step"], mode="lines", name="Reward per step (raw)", opacity=0.3))
fig5.add_trace(go.Scatter(x=df_train["episode"], y=df_train["reward_per_step_smoothed"], mode="lines", name="Reward per step (smoothed)"))
fig5.update_layout(title={"text": "Mean reward per step during training (4-DOF)<br><span style='font-size: 18px; font-weight: normal;'>Normalizes for episode length to show balancing quality</span>"})
fig5.update_xaxes(title_text="Episode (during training)")
fig5.update_yaxes(title_text="Reward per step")
fig5.write_image(os.path.join(output_dir, "training_reward_per_step_4dof.png"))
with open(os.path.join(output_dir, "training_reward_per_step_4dof.png.meta.json"), "w") as f:
    json.dump({"caption": "Mean reward per step per episode during 4-DOF PPO training",
               "description": "Line chart showing total reward divided by episode length."}, f)