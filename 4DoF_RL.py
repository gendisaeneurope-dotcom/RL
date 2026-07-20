import os
output_dir = "./output_4dof/"
os.makedirs(output_dir, exist_ok=True)

import math
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import load_results

# ── Re-declare the same custom env used in training ──
import gymnasium as gym
from gymnasium import spaces

xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ankle_hip.xml")
_base_env = gym.make("InvertedPendulum-v5", xml_file=xml_path).unwrapped
AnkleHipEnv = type(_base_env)

class MyAnkleHipEnv(AnkleHipEnv):
    def __init__(self, **kwargs):
        super().__init__(xml_file=xml_path, **kwargs)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        self._current_step = 0
        self._max_steps = 1000

    def reset(self, **kwargs):
        self._current_step = 0
        return super().reset(**kwargs)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()

        ankle_angle = observation[0]
        hip_angle = observation[1]

        failed = bool(
            not np.isfinite(observation).all()
            or (np.abs(ankle_angle) > 0.5)
            or (np.abs(hip_angle) > 0.5)
        )

        h_ankle = np.cos(ankle_angle)
        h_hip = np.cos(hip_angle)
        h = 0.5 * h_ankle + 0.5 * h_hip

        self._current_step += 1
        success = (self._current_step >= self._max_steps) and not failed

        if success:
            reward = 1000.0
            terminated = True
        elif failed:
            reward = -100.0 - 400.0 * (1.0 - h)
            terminated = True
        else:
            ankle_effort = float(action[0]) ** 2
            hip_effort = float(action[1]) ** 2
            omega = 0.5
            reward = h - omega * (ankle_effort + hip_effort)
            terminated = False

        info = {"reward_survive": reward}
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info


# ── Load trained model ──
model = PPO.load("ppo_ankle_hip")

# ── Roll out 20 evaluation episodes ──
env = TimeLimit(MyAnkleHipEnv(), max_episode_steps=1000)

rows = []
for ep in range(1, 21):
    obs, info = env.reset()
    total_reward = 0.0
    done = False
    step_idx = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        ankle_deg = float(obs[0]) * 180 / math.pi
        hip_deg = float(obs[1]) * 180 / math.pi
        ankle_act = float(action[0])
        hip_act = float(action[1])

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        step_idx += 1

        rows.append({
            "episode": ep,
            "step": step_idx,
            "ankle_deg": ankle_deg,
            "hip_deg": hip_deg,
            "ankle_action": ankle_act,
            "hip_action": hip_act,
            "reward": float(reward),
            "cum_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated
        })

        done = terminated or truncated

env.close()

df = pd.DataFrame(rows)
df.to_csv(os.path.join(output_dir, "ankle_hip_trajectories.csv"), index=False)

summary = df.groupby("episode").agg(
    total_reward=("reward", "sum"),
    steps=("step", "max"),
    mean_abs_ankle_deg=("ankle_deg", lambda s: np.mean(np.abs(s))),
    mean_abs_hip_deg=("hip_deg", lambda s: np.mean(np.abs(s))),
    max_abs_ankle_deg=("ankle_deg", lambda s: np.max(np.abs(s))),
    max_abs_hip_deg=("hip_deg", lambda s: np.max(np.abs(s))),
).reset_index()

summary.to_csv(os.path.join(output_dir, "ankle_hip_episode_summary.csv"), index=False)
print(summary)


# ── Plot 1: total reward per episode ──
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=summary["episode"], y=summary["total_reward"],
    mode="lines+markers", name="Reward"
))
fig1.update_layout(title={
    "text": "Episode reward (20 eval eps, 4-DOF)<br><span style='font-size: 18px; font-weight: normal;'>Ankle-hip model under new shaping reward</span>"
})
fig1.update_xaxes(title_text="Episode")
fig1.update_yaxes(title_text="Reward")
fig1.write_image(os.path.join(output_dir, "episode_rewards_4dof.png"))

with open(os.path.join(output_dir, "episode_rewards_4dof.png.meta.json"), "w") as f:
    json.dump({
        "caption": "Episode rewards over 20 evaluation episodes (4-DOF ankle-hip)",
        "description": "Line chart of total reward for each evaluation episode under the ankle-hip shaping reward."
    }, f)


# ── Plot 2: ankle + hip angle trajectories, episode 1 ──
d = df[df["episode"] == 1]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=d["step"], y=d["ankle_deg"], mode="lines", name="Ankle angle"))
fig2.add_trace(go.Scatter(x=d["step"], y=d["hip_deg"], mode="lines", name="Hip angle"))
fig2.update_layout(title={
    "text": "Joint angle trajectory (episode 1, 4-DOF)<br><span style='font-size: 18px; font-weight: normal;'>Recovery-then-hold behavior for both joints</span>"
})
fig2.update_xaxes(title_text="Step")
fig2.update_yaxes(title_text="Angle (deg)")
fig2.write_image(os.path.join(output_dir, "joint_angle_trajectories_4dof.png"))

with open(os.path.join(output_dir, "joint_angle_trajectories_4dof.png.meta.json"), "w") as f:
    json.dump({
        "caption": "Ankle and hip angle trajectories for episode 1",
        "description": "Line chart showing ankle and hip joint angles in degrees across timesteps."
    }, f)


# ── Plot 3: training reward curve ──
df_train = load_results("./training_logs_4dof/")
df_train["episode"] = range(len(df_train))
df_train["r_smoothed"] = df_train["r"].rolling(20, min_periods=1).mean()

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df_train["episode"], y=df_train["r"], mode="lines", name="Raw reward", opacity=0.3))
fig3.add_trace(go.Scatter(x=df_train["episode"], y=df_train["r_smoothed"], mode="lines", name="Smoothed (20-ep rolling mean)"))
fig3.update_layout(title={
    "text": "Training reward per episode (4-DOF)<br><span style='font-size: 18px; font-weight: normal;'>Reward should trend toward the success bonus as policy learns</span>"
})
fig3.update_xaxes(title_text="Episode (during training)")
fig3.update_yaxes(title_text="Reward")
fig3.write_image(os.path.join(output_dir, "training_reward_4dof.png"))

with open(os.path.join(output_dir, "training_reward_4dof.png.meta.json"), "w") as f:
    json.dump({
        "caption": "Reward per episode during 4-DOF PPO training",
        "description": "Line chart showing raw and smoothed reward across training episodes for the ankle-hip model."
    }, f)


# ── Plot 4: episode length during training ──
df_train["l_smoothed"] = df_train["l"].rolling(20, min_periods=1).mean()

fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=df_train["episode"], y=df_train["l"], mode="lines", name="Episode length (raw)", opacity=0.3))
fig4.add_trace(go.Scatter(x=df_train["episode"], y=df_train["l_smoothed"], mode="lines", name="Episode length (smoothed)"))
fig4.update_layout(title={
    "text": "Episode length during training (4-DOF)<br><span style='font-size: 18px; font-weight: normal;'>Longer episodes reflect improved balance duration</span>"
})
fig4.update_xaxes(title_text="Episode (during training)")
fig4.update_yaxes(title_text="Steps per episode")
fig4.write_image(os.path.join(output_dir, "training_episode_length_4dof.png"))

with open(os.path.join(output_dir, "training_episode_length_4dof.png.meta.json"), "w") as f:
    json.dump({
        "caption": "Episode length per episode during 4-DOF PPO training",
        "description": "Line chart of raw and smoothed episode length across training episodes."
    }, f)


# ── Plot 5: mean reward per step during training ──
df_train["reward_per_step"] = df_train["r"] / df_train["l"]
df_train["reward_per_step_smoothed"] = df_train["reward_per_step"].rolling(20, min_periods=1).mean()

fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=df_train["episode"], y=df_train["reward_per_step"], mode="lines", name="Reward per step (raw)", opacity=0.3))
fig5.add_trace(go.Scatter(x=df_train["episode"], y=df_train["reward_per_step_smoothed"], mode="lines", name="Reward per step (smoothed)"))
fig5.update_layout(title={
    "text": "Mean reward per step during training (4-DOF)<br><span style='font-size: 18px; font-weight: normal;'>Normalizes for episode length to show balancing quality</span>"
})
fig5.update_xaxes(title_text="Episode (during training)")
fig5.update_yaxes(title_text="Reward per step")
fig5.write_image(os.path.join(output_dir, "training_reward_per_step_4dof.png"))

with open(os.path.join(output_dir, "training_reward_per_step_4dof.png.meta.json"), "w") as f:
    json.dump({
        "caption": "Mean reward per step per episode during 4-DOF PPO training",
        "description": "Line chart showing total reward divided by episode length, normalizing for episode duration."
    }, f)