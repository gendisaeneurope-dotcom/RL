import os
output_dir = "./output_4dof_robust/"
os.makedirs(output_dir, exist_ok=True)

import math
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO


xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ankle_hip.xml")
_base_env = gym.make("InvertedPendulum-v5", xml_file=xml_path).unwrapped
AnkleHipEnv = type(_base_env)


class MyAnkleHipEnvRobust(AnkleHipEnv):
    def __init__(self, disturb_prob=0.05, force_range=(-20, 20), **kwargs):
        super().__init__(xml_file=xml_path, **kwargs)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        self._current_step = 0
        self._max_steps = 1000
        self.disturb_prob = disturb_prob
        self.force_range = force_range

    def reset(self, **kwargs):
        self._current_step = 0
        return super().reset(**kwargs)

    def step(self, action):
        if np.random.rand() < self.disturb_prob:
            force = np.random.uniform(*self.force_range)
            self.data.xfrc_applied[1, 0] = force
        else:
            self.data.xfrc_applied[1, 0] = 0.0

        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()

        ankle_angle = observation[0]
        hip_angle = observation[1]

        failed = bool(
            not np.isfinite(observation).all()
            or (np.abs(ankle_angle) > 0.15)
            or (np.abs(hip_angle) > 0.15)
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
            omega = 0.1
            reward = h - omega * (ankle_effort + hip_effort)
            terminated = False

        info = {"reward_survive": reward}
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info


def run_eval(model_path, seed_offset, n_episodes=20):
    model = PPO.load(model_path)
    env = TimeLimit(MyAnkleHipEnvRobust(), max_episode_steps=1000)

    rows = []
    for ep in range(1, n_episodes + 1):
        np.random.seed(seed_offset + ep)
        obs, info = env.reset(seed=seed_offset + ep)
        total_reward = 0.0
        done = False
        step_idx = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            ankle_deg = float(obs[0]) * 180 / math.pi
            hip_deg = float(obs[1]) * 180 / math.pi

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            step_idx += 1

            rows.append({
                "episode": ep, "step": step_idx,
                "ankle_deg": ankle_deg, "hip_deg": hip_deg,
                "reward": float(reward), "cum_reward": total_reward,
                "terminated": terminated, "truncated": truncated
            })
            done = terminated or truncated

    env.close()
    return pd.DataFrame(rows)


# ── Same disturbance seeds for both models: fair comparison ──
df_baseline = run_eval("ppo_ankle_hip", seed_offset=1000)
df_robust = run_eval("ppo_ankle_hip_robust", seed_offset=1000)

df_baseline["model"] = "Baseline (no disturbance training)"
df_robust["model"] = "Robust (disturbance training)"
df_all = pd.concat([df_baseline, df_robust], ignore_index=True)
df_all.to_csv(os.path.join(output_dir, "robust_comparison_trajectories.csv"), index=False)

summary = df_all.groupby(["model", "episode"]).agg(
    total_reward=("reward", "sum"),
    steps=("step", "max"),
    max_abs_ankle_deg=("ankle_deg", lambda s: np.max(np.abs(s))),
    max_abs_hip_deg=("hip_deg", lambda s: np.max(np.abs(s))),
).reset_index()
summary.to_csv(os.path.join(output_dir, "robust_comparison_summary.csv"), index=False)

overall = summary.groupby("model").agg(
    mean_reward=("total_reward", "mean"),
    std_reward=("total_reward", "std"),
    mean_steps=("steps", "mean"),
    survival_rate=("steps", lambda s: np.mean(s >= 1000)),
).reset_index()
overall.to_csv(os.path.join(output_dir, "robust_comparison_overall.csv"), index=False)
print(overall)


# ── Plot 1: reward per episode, both models ──
fig1 = go.Figure()
for m in summary["model"].unique():
    d = summary[summary["model"] == m]
    fig1.add_trace(go.Scatter(x=d["episode"], y=d["total_reward"], mode="lines+markers", name=m))
fig1.update_layout(title={"text": "Episode reward under disturbance (20 eps)<br><span style='font-size: 18px; font-weight: normal;'>Baseline vs. disturbance-trained policy</span>"})
fig1.update_xaxes(title_text="Episode")
fig1.update_yaxes(title_text="Reward")
fig1.write_image(os.path.join(output_dir, "robust_reward_comparison.png"))
with open(os.path.join(output_dir, "robust_reward_comparison.png.meta.json"), "w") as f:
    json.dump({"caption": "Episode reward comparison under disturbance", "description": "Reward per episode for baseline and robust models under identical random disturbances."}, f)


# ── Plot 2: episode length (survival), both models ──
fig2 = go.Figure()
for m in summary["model"].unique():
    d = summary[summary["model"] == m]
    fig2.add_trace(go.Scatter(x=d["episode"], y=d["steps"], mode="lines+markers", name=m))
fig2.update_layout(title={"text": "Episode length under disturbance (20 eps)<br><span style='font-size: 18px; font-weight: normal;'>Higher = more robust to random pushes</span>"})
fig2.update_xaxes(title_text="Episode")
fig2.update_yaxes(title_text="Steps survived")
fig2.write_image(os.path.join(output_dir, "robust_survival_comparison.png"))
with open(os.path.join(output_dir, "robust_survival_comparison.png.meta.json"), "w") as f:
    json.dump({"caption": "Survival length comparison under disturbance", "description": "Episode length for baseline and robust models under identical random disturbances."}, f)


# ── Plot 3: joint angle trajectories, episode 1, both models ──
fig3 = go.Figure()
for m in df_all["model"].unique():
    d = df_all[(df_all["model"] == m) & (df_all["episode"] == 1)]
    fig3.add_trace(go.Scatter(x=d["step"], y=d["ankle_deg"], mode="lines", name=f"{m} - ankle"))
    fig3.add_trace(go.Scatter(x=d["step"], y=d["hip_deg"], mode="lines", name=f"{m} - hip"))
fig3.update_layout(title={"text": "Joint angles under disturbance (episode 1)<br><span style='font-size: 18px; font-weight: normal;'>Baseline vs. robust policy behavior</span>"})
fig3.update_xaxes(title_text="Step")
fig3.update_yaxes(title_text="Angle (deg)")
fig3.write_image(os.path.join(output_dir, "robust_angle_comparison.png"))
with open(os.path.join(output_dir, "robust_angle_comparison.png.meta.json"), "w") as f:
    json.dump({"caption": "Joint angle trajectories comparison under disturbance", "description": "Ankle and hip angles for baseline vs. robust-trained policy under the same disturbance sequence."}, f)