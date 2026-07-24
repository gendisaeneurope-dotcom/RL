import os
output_dir = "./output_4dof_target/"
os.makedirs(output_dir, exist_ok=True)

import math
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from RL_4DoF_target import My4DOFTargetEnv, JOINT_NAMES, N_JOINTS
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import load_results

# NOTE: rename to match whatever you actually saved -- check your working
# directory / saved .zip filename, don't just assume this matches.
model = PPO.load("ppo_ankle_hip_target")

env = TimeLimit(My4DOFTargetEnv(disturb_prob=0.3), max_episode_steps=1000)

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
            "com_y": float(info["com_y"]), "target_y": float(info["target_y"]),
            "terminated": terminated, "truncated": truncated
        })
        rows.append(row)

        done = terminated or truncated

env.close()

df = pd.DataFrame(rows)
df.to_csv(os.path.join(output_dir, "4dof_target_trajectories.csv"), index=False)

agg_dict = {
    "total_reward": ("reward", "sum"), "steps": ("step", "max"),
    "target_y": ("target_y", "first"),
    "final_com_y": ("com_y", "last"),
}
for name in JOINT_NAMES:
    agg_dict[f"mean_abs_{name}_deg"] = (f"{name}_deg", lambda s: np.mean(np.abs(s)))
    agg_dict[f"max_abs_{name}_deg"] = (f"{name}_deg", lambda s: np.max(np.abs(s)))
summary = df.groupby("episode").agg(**agg_dict).reset_index()
summary["final_error"] = summary["final_com_y"] - summary["target_y"]
summary.to_csv(os.path.join(output_dir, "4dof_target_episode_summary.csv"), index=False)
print(summary)


# ── Core plot: does com_y actually track target_y over time, for several
#    episodes with different targets? This is the direct visual version of
#    the printed error check. ──
n_show = min(6, df["episode"].nunique())
fig_track = make_subplots(rows=n_show, cols=1, shared_xaxes=True,
                          subplot_titles=[f"Episode {ep}" for ep in range(1, n_show + 1)])
for i, ep in enumerate(range(1, n_show + 1)):
    d = df[df["episode"] == ep]
    fig_track.add_trace(go.Scatter(x=d["step"], y=d["com_y"], mode="lines", name="CoM y",
                                    line=dict(color="royalblue"), showlegend=(i == 0)), row=i + 1, col=1)
    fig_track.add_trace(go.Scatter(x=d["step"], y=d["target_y"], mode="lines", name="Target y",
                                    line=dict(color="firebrick", dash="dash"), showlegend=(i == 0)), row=i + 1, col=1)
fig_track.update_layout(
    height=180 * n_show,
    title={"text": "CoM-y tracking vs. target-y, across episodes with different targets<br>"
                   "<span style='font-size:14px;font-weight:normal;'>Flat line matching the dashed target = good tracking; flat line ignoring target = old failure mode</span>"}
)
fig_track.update_yaxes(title_text="y (m)")
fig_track.update_xaxes(title_text="Step", row=n_show, col=1)
fig_track.write_image(os.path.join(output_dir, "com_target_tracking.png"))
with open(os.path.join(output_dir, "com_target_tracking.png.meta.json"), "w") as f:
    json.dump({
        "caption": "CoM-y vs. target-y over time, for several evaluation episodes with different sampled targets",
        "description": "Stacked line charts, one per episode, each showing CoM-y (solid) against its episode's fixed target-y (dashed)."
    }, f)


# ── Scatter: final com_y vs target_y across all 20 eval episodes.
#    Perfect tracking = points on the y=x line. ──
fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(x=summary["target_y"], y=summary["final_com_y"], mode="markers",
                                  name="Episodes", marker=dict(size=9)))
lims = [summary["target_y"].min(), summary["target_y"].max()]
fig_scatter.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="Perfect tracking (y=x)",
                                  line=dict(color="gray", dash="dot")))
fig_scatter.update_layout(title={"text": "Final CoM-y vs. target-y (20 eval episodes)<br>"
                                          "<span style='font-size:14px;font-weight:normal;'>Points on the dotted line = perfect tracking</span>"})
fig_scatter.update_xaxes(title_text="Target y (m)")
fig_scatter.update_yaxes(title_text="Final CoM y (m)")
fig_scatter.write_image(os.path.join(output_dir, "target_tracking_scatter.png"))
with open(os.path.join(output_dir, "target_tracking_scatter.png.meta.json"), "w") as f:
    json.dump({
        "caption": "Final CoM-y position vs. sampled target-y for 20 evaluation episodes",
        "description": "Scatter plot of final CoM-y against target-y, with a reference y=x line indicating perfect tracking."
    }, f)


# ── Joint angles, episode 1 ──
d = df[df["episode"] == 1]
fig_joints = go.Figure()
for name in JOINT_NAMES:
    fig_joints.add_trace(go.Scatter(x=d["step"], y=d[f"{name}_deg"], mode="lines", name=f"{name} angle"))
fig_joints.update_layout(title={"text": "Joint angle trajectories (episode 1, target reward)"})
fig_joints.update_xaxes(title_text="Step")
fig_joints.update_yaxes(title_text="Angle (deg)")
fig_joints.write_image(os.path.join(output_dir, "joint_angle_trajectories_target.png"))
with open(os.path.join(output_dir, "joint_angle_trajectories_target.png.meta.json"), "w") as f:
    json.dump({"caption": "Joint angle trajectories for episode 1 under the target-reaching reward",
               "description": "Line chart showing all 4 joint angles across timesteps."}, f)


# ── Episode reward summary ──
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=summary["episode"], y=summary["total_reward"], mode="lines+markers", name="Reward"))
fig1.update_layout(title={"text": "Episode reward (20 eval eps, target reward candidate)"})
fig1.update_xaxes(title_text="Episode")
fig1.update_yaxes(title_text="Reward")
fig1.write_image(os.path.join(output_dir, "episode_rewards_target.png"))
with open(os.path.join(output_dir, "episode_rewards_target.png.meta.json"), "w") as f:
    json.dump({"caption": "Episode rewards over 20 evaluation episodes (target-reaching reward)",
               "description": "Line chart of total reward for each evaluation episode."}, f)


# ── Training curves ──
df_train = load_results("./training_logs_4dof_target/")
df_train["episode"] = range(len(df_train))
df_train["r_smoothed"] = df_train["r"].rolling(20, min_periods=1).mean()

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df_train["episode"], y=df_train["r"], mode="lines", name="Raw reward", opacity=0.3))
fig3.add_trace(go.Scatter(x=df_train["episode"], y=df_train["r_smoothed"], mode="lines", name="Smoothed (20-ep rolling mean)"))
fig3.update_layout(title={"text": "Training reward per episode (target reward candidate)"})
fig3.update_xaxes(title_text="Episode (during training)")
fig3.update_yaxes(title_text="Reward")
fig3.write_image(os.path.join(output_dir, "training_reward_target.png"))
with open(os.path.join(output_dir, "training_reward_target.png.meta.json"), "w") as f:
    json.dump({"caption": "Reward per episode during target-reward PPO training",
               "description": "Line chart showing raw and smoothed reward across training episodes."}, f)

df_train["reward_per_step"] = df_train["r"] / df_train["l"]
df_train["reward_per_step_smoothed"] = df_train["reward_per_step"].rolling(20, min_periods=1).mean()
fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=df_train["episode"], y=df_train["reward_per_step"], mode="lines", name="Reward per step (raw)", opacity=0.3))
fig5.add_trace(go.Scatter(x=df_train["episode"], y=df_train["reward_per_step_smoothed"], mode="lines", name="Reward per step (smoothed)"))
fig5.update_layout(title={"text": "Mean reward per step during training (target reward candidate)"})
fig5.update_xaxes(title_text="Episode (during training)")
fig5.update_yaxes(title_text="Reward per step")
fig5.write_image(os.path.join(output_dir, "training_reward_per_step_target.png"))
with open(os.path.join(output_dir, "training_reward_per_step_target.png.meta.json"), "w") as f:
    json.dump({"caption": "Mean reward per step per episode during target-reward PPO training",
               "description": "Line chart showing total reward divided by episode length."}, f)
