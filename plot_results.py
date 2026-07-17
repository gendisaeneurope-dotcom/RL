import math
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from gymnasium.envs.mujoco.inverted_pendulum_v5 import InvertedPendulumEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import load_results


# ── Same custom env as before ──
class MyInvertedPendulumEnv(InvertedPendulumEnv):
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        terminated = bool(not np.isfinite(observation).all() or (np.abs(observation[1]) > 0.2))

        com_deviation = observation[1] ** 2
        control_effort = float(action[0]) ** 2
        reward = -(1.0 * com_deviation + 0.1 * control_effort) if not terminated else -10.0

        info = {"reward_survive": reward}
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info


# ── Load trained model ──
model = PPO.load("ppo_inverted_pendulum_custom")

# ── Roll out 20 evaluation episodes and store trajectories ──
env = TimeLimit(MyInvertedPendulumEnv(), max_episode_steps=1000)

rows = []
for ep in range(1, 21):
    obs, info = env.reset()
    total_reward = 0.0
    done = False
    step_idx = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        angle_rad = float(obs[1])
        angle_deg = angle_rad * 180 / math.pi
        act = float(action[0])

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        step_idx += 1

        rows.append({
            "episode": ep,
            "step": step_idx,
            "angle_rad": angle_rad,
            "angle_deg": angle_deg,
            "action": act,
            "reward": float(reward),
            "cum_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated
        })

        done = terminated or truncated

env.close()

# ── Save raw trajectory table ──
df = pd.DataFrame(rows)
df.to_csv("custom_reward_trajectories.csv", index=False)

# ── Per-episode summary table ──
summary = df.groupby("episode").agg(
    total_reward=("reward", "sum"),
    steps=("step", "max"),
    mean_abs_angle_deg=("angle_deg", lambda s: np.mean(np.abs(s))),
    max_abs_angle_deg=("angle_deg", lambda s: np.max(np.abs(s))),
    mean_abs_action=("action", lambda s: np.mean(np.abs(s))),
).reset_index()

summary.to_csv("custom_reward_episode_summary.csv", index=False)

print(summary)

# ── Plot 1: total reward per episode ──
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=summary["episode"],
    y=summary["total_reward"],
    mode="lines+markers",
    name="Reward"
))
fig1.update_layout(
    title={
        "text": "Episode reward (20 eval eps)<br><span style='font-size: 18px; font-weight: normal;'>Custom reward stays near zero and highly consistent</span>"
    }
)
fig1.update_xaxes(title_text="Episode")
fig1.update_yaxes(title_text="Reward")
fig1.write_image("episode_rewards.png")

with open("episode_rewards.png.meta.json", "w") as f:
    json.dump({
        "caption": "Episode rewards over 20 evaluation episodes",
        "description": "Line chart of total reward for each evaluation episode under the custom reward."
    }, f)

# ── Plot 2: pole angle trajectories for first 3 episodes ──
fig2 = go.Figure()
for ep in [1, 2, 3]:
    d = df[df["episode"] == ep]
    fig2.add_trace(go.Scatter(
        x=d["step"],
        y=d["angle_deg"],
        mode="lines",
        name=f"Ep {ep}"
    ))

fig2.update_layout(
    title={
        "text": "Pole angle trajectory (episodes 1-3)<br><span style='font-size: 18px; font-weight: normal;'>Angle remains extremely close to zero throughout</span>"
    }
)
fig2.update_xaxes(title_text="Step")
fig2.update_yaxes(title_text="Angle (deg)")
fig2.write_image("pole_angle_trajectories.png")

with open("pole_angle_trajectories.png.meta.json", "w") as f:
    json.dump({
        "caption": "Pole angle trajectories for episodes 1 to 3",
        "description": "Line chart showing pole angle in degrees across timesteps for the first three evaluation episodes."
    }, f)

# ── Plot 0: training reward curve (requires Monitor wrapper used during training) ──
df_train = load_results("./training_logs/")
df_train["episode"] = range(len(df_train))
df_train["r_smoothed"] = df_train["r"].rolling(20, min_periods=1).mean()

fig0 = go.Figure()
fig0.add_trace(go.Scatter(
    x=df_train["episode"], y=df_train["r"],
    mode="lines", name="Raw reward", opacity=0.3
))
fig0.add_trace(go.Scatter(
    x=df_train["episode"], y=df_train["r_smoothed"],
    mode="lines", name="Smoothed (20-ep rolling mean)"
))
fig0.update_layout(
    title={
        "text": "Training reward per episode<br><span style='font-size: 18px; font-weight: normal;'>Reward should trend toward zero as the policy learns</span>"
    }
)
fig0.update_xaxes(title_text="Episode (during training)")
fig0.update_yaxes(title_text="Reward")
fig0.write_image("training_reward.png")

with open("training_reward.png.meta.json", "w") as f:
    json.dump({
        "caption": "Reward per episode during PPO training",
        "description": "Line chart showing raw and smoothed reward across episodes encountered while training, not evaluation."
    }, f)

# ── Plot 0b: episode length and per-step reward during training ──
df_train["reward_per_step"] = df_train["r"] / df_train["l"]
df_train["l_smoothed"] = df_train["l"].rolling(20, min_periods=1).mean()
df_train["reward_per_step_smoothed"] = df_train["reward_per_step"].rolling(20, min_periods=1).mean()

fig0b = go.Figure()
fig0b.add_trace(go.Scatter(
    x=df_train["episode"], y=df_train["l"],
    mode="lines", name="Episode length (raw)", opacity=0.3
))
fig0b.add_trace(go.Scatter(
    x=df_train["episode"], y=df_train["l_smoothed"],
    mode="lines", name="Episode length (smoothed)"
))
fig0b.update_layout(
    title={
        "text": "Episode length during training<br><span style='font-size: 18px; font-weight: normal;'>Longer episodes may explain a more negative total reward</span>"
    }
)
fig0b.update_xaxes(title_text="Episode (during training)")
fig0b.update_yaxes(title_text="Steps per episode")
fig0b.write_image("training_episode_length.png")

with open("training_episode_length.png.meta.json", "w") as f:
    json.dump({
        "caption": "Episode length per episode during PPO training",
        "description": "Line chart of raw and smoothed episode length across training episodes."
    }, f)

fig0c = go.Figure()
fig0c.add_trace(go.Scatter(
    x=df_train["episode"], y=df_train["reward_per_step"],
    mode="lines", name="Reward per step (raw)", opacity=0.3
))
fig0c.add_trace(go.Scatter(
    x=df_train["episode"], y=df_train["reward_per_step_smoothed"],
    mode="lines", name="Reward per step (smoothed)"
))
fig0c.update_layout(
    title={
        "text": "Mean reward per step during training<br><span style='font-size: 18px; font-weight: normal;'>Normalizes for episode length to show true balancing quality</span>"
    }
)
fig0c.update_xaxes(title_text="Episode (during training)")
fig0c.update_yaxes(title_text="Reward per step")
fig0c.write_image("training_reward_per_step.png")

with open("training_reward_per_step.png.meta.json", "w") as f:
    json.dump({
        "caption": "Mean reward per step per episode during PPO training",
        "description": "Line chart showing total reward divided by episode length, normalizing for episode duration."
    }, f)