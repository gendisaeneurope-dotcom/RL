import os
output_dir = "./output_4dof/"
os.makedirs(output_dir, exist_ok=True)

import math
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from RL_4DoF import My4DOFEnv, JOINT_NAMES, N_JOINTS
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import load_results

model = PPO.load("ppo_ankle_hip")

env = TimeLimit(My4DOFEnv(), max_episode_steps=1000)

rows = []
for ep in range(1, 21):
    obs, info = env.reset()
    total_reward = 0.0
    done = False
    step_idx = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        # CoM position, read directly from MuJoCo -- independent of the reward
        # function's internal math.
        com = env.unwrapped.data.subtree_com[env.unwrapped.root_body_id][:2].copy()

        row = {"episode": ep, "step": step_idx + 1, "com_x": float(com[0]), "com_y": float(com[1])}
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

# Full per-step trajectory data (all episodes, all joints, CoM, reward) --
# used both as raw data for your records and as the source for every plot below.
df.to_csv(os.path.join(output_dir, "ankle_hip_trajectories.csv"), index=False)

agg_dict = {
    "total_reward": ("reward", "sum"), "steps": ("step", "max"),
    "mean_abs_com_x": ("com_x", lambda s: np.mean(np.abs(s))),
    "max_abs_com_x": ("com_x", lambda s: np.max(np.abs(s))),
}
for name in JOINT_NAMES:
    agg_dict[f"mean_abs_{name}_deg"] = (f"{name}_deg", lambda s: np.mean(np.abs(s)))
    agg_dict[f"max_abs_{name}_deg"] = (f"{name}_deg", lambda s: np.max(np.abs(s)))
summary = df.groupby("episode").agg(**agg_dict).reset_index()
print(summary)

# Per-episode summary: total reward, episode length, and mean/max joint angle
# magnitude per joint -- one row per episode, useful for quick comparisons
# across the 20 eval episodes without needing the full per-step data.
summary.to_csv(os.path.join(output_dir, "ankle_hip_episode_summary.csv"), index=False)


# PLOT: joint angles (top) + CoM x/y position (bottom), stacked, episode 1.
# WHY: answers "why are the joints moving if the goal is CoM-at-0"
# to see directly whether joint motion is active corrective sway around a held
# CoM, or whether it settles into a fixed nonzero posture instead.
d = df[df["episode"] == 1]
fig_combined = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    subplot_titles=("Joint angle trajectories (episode 1)", "CoM position (episode 1)")
)
for name in JOINT_NAMES:
    fig_combined.add_trace(go.Scatter(x=d["step"], y=d[f"{name}_deg"], mode="lines", name=f"{name} angle"), row=1, col=1)

fig_combined.add_trace(go.Scatter(x=d["step"], y=d["com_x"], mode="lines", name="CoM x"), row=2, col=1)
fig_combined.add_trace(go.Scatter(x=d["step"], y=d["com_y"], mode="lines", name="CoM y"), row=2, col=1)

fig_combined.update_layout(
    title={"text": "Joint angles vs. CoM position (episode 1)<br>"
                   "<span style='font-size:16px;font-weight:normal;'>"
                   "Checks whether joint motion is actually corrective sway around a held CoM, or drift</span>"},
    height=700,
)
fig_combined.update_yaxes(title_text="Angle (deg)", row=1, col=1)
fig_combined.update_yaxes(title_text="CoM position (m)", row=2, col=1)
fig_combined.update_xaxes(title_text="Step", row=2, col=1)
fig_combined.write_image(os.path.join(output_dir, "joint_angles_vs_com.png"))
with open(os.path.join(output_dir, "joint_angles_vs_com.png.meta.json"), "w") as f:
    json.dump({
        "caption": "Joint angle trajectories and CoM x/y position for episode 1",
        "description": "Two stacked line charts: joint angles over time (top), CoM position over time (bottom), same episode/timesteps."
    }, f)


# PLOT: joint angles only, episode 1 (same data as the top half of the combined
# plot above, but on its own. to show joint behavior
# without also displaying CoM, e.g. in a slide focused on movement pattern).
fig_joints = go.Figure()
for name in JOINT_NAMES:
    fig_joints.add_trace(go.Scatter(x=d["step"], y=d[f"{name}_deg"], mode="lines", name=f"{name.capitalize()} angle"))
fig_joints.update_layout(title={"text": "Joint angle trajectories (episode 1, 4-DOF)<br>"
                                         "<span style='font-size: 18px; font-weight: normal;'>Recovery-then-hold behavior across all 4 joints</span>"})
fig_joints.update_xaxes(title_text="Step")
fig_joints.update_yaxes(title_text="Angle (deg)")
fig_joints.write_image(os.path.join(output_dir, "joint_angle_trajectories_4dof.png"))
with open(os.path.join(output_dir, "joint_angle_trajectories_4dof.png.meta.json"), "w") as f:
    json.dump({"caption": "Ankle and hip joint angle trajectories for episode 1",
               "description": "Line chart showing all 4 joint angles across timesteps."}, f)


# PLOT: total reward per evaluation episode, across the 20 eval rollouts.
# WHY: quick check for consistency across episodes -- flat/tight = reliable
# policy, scattered = performance varies a lot depending on initial conditions
# or disturbance timing.
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=summary["episode"], y=summary["total_reward"], mode="lines+markers", name="Reward"))
fig1.update_layout(title={"text": "Episode reward (20 eval eps, 4-DOF)"})
fig1.update_xaxes(title_text="Episode")
fig1.update_yaxes(title_text="Reward")
fig1.write_image(os.path.join(output_dir, "episode_rewards_4dof.png"))
with open(os.path.join(output_dir, "episode_rewards_4dof.png.meta.json"), "w") as f:
    json.dump({"caption": "Episode rewards over 20 evaluation episodes",
               "description": "Line chart of total reward for each evaluation episode."}, f)


df_train = load_results("./training_logs_4dof/")
df_train["episode"] = range(len(df_train))
df_train["r_smoothed"] = df_train["r"].rolling(20, min_periods=1).mean()

# PLOT: raw + smoothed reward per episode DURING training (not eval).
# WHY: shows how learning progressed over the whole training run -- watch for
# a flat/near-zero period followed by a sudden jump (a red flag for reward
# shaping problems) vs. a smooth, gradual climb (healthy learning).
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df_train["episode"], y=df_train["r"], mode="lines", name="Raw reward", opacity=0.3))
fig3.add_trace(go.Scatter(x=df_train["episode"], y=df_train["r_smoothed"], mode="lines", name="Smoothed (20-ep rolling mean)"))
fig3.update_layout(title={"text": "Training reward per episode (4-DOF)"})
fig3.update_xaxes(title_text="Episode (during training)")
fig3.update_yaxes(title_text="Reward")
fig3.write_image(os.path.join(output_dir, "training_reward_4dof.png"))
with open(os.path.join(output_dir, "training_reward_4dof.png.meta.json"), "w") as f:
    json.dump({"caption": "Reward per episode during 4-DOF PPO training",
               "description": "Line chart showing raw and smoothed reward across training episodes."}, f)


# PLOT: episode length DURING training -- how many steps survived before
# termination, per training episode.
# WHY: total reward naturally rises just because episodes get longer early in
# training; this isolates that effect and shows exactly when the policy
# stopped dying early and started reaching the full episode length.
df_train["l_smoothed"] = df_train["l"].rolling(20, min_periods=1).mean()
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=df_train["episode"], y=df_train["l"], mode="lines", name="Episode length (raw)", opacity=0.3))
fig4.add_trace(go.Scatter(x=df_train["episode"], y=df_train["l_smoothed"], mode="lines", name="Episode length (smoothed)"))
fig4.update_layout(title={"text": "Episode length during training (4-DOF)<br>"
                                   "<span style='font-size: 18px; font-weight: normal;'>Longer episodes reflect improved balance duration</span>"})
fig4.update_xaxes(title_text="Episode (during training)")
fig4.update_yaxes(title_text="Steps per episode")
fig4.write_image(os.path.join(output_dir, "training_episode_length_4dof.png"))
with open(os.path.join(output_dir, "training_episode_length_4dof.png.meta.json"), "w") as f:
    json.dump({"caption": "Episode length per episode during 4-DOF PPO training",
               "description": "Line chart of raw and smoothed episode length across training episodes."}, f)


# PLOT: reward per step (total reward / episode length) DURING training.
# WHY: strips out the "episodes got longer" effect from the raw reward curve
# above, so this shows whether the QUALITY of balance/behavior is actually
# improving, not just how long the policy survives.
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