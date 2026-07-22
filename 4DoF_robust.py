import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from gymnasium import spaces
import numpy as np


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


# ── Training ──
log_dir = "./training_logs_4dof_robust/"
os.makedirs(log_dir, exist_ok=True)

env = TimeLimit(MyAnkleHipEnvRobust(), max_episode_steps=1000)
env = Monitor(env, log_dir)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb_logs_4dof_robust/")
model.learn(total_timesteps=500_000)
model.save("ppo_ankle_hip_robust")
env.close()


# ── Quick visual check ──
model = PPO.load("ppo_ankle_hip_robust")
env = TimeLimit(MyAnkleHipEnvRobust(render_mode="human"), max_episode_steps=1000)
obs, info = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()


# ── Quantitative evaluation under disturbance ──
eval_env = TimeLimit(MyAnkleHipEnvRobust(), max_episode_steps=1000)

episode_rewards, episode_lengths = evaluate_policy(
    model, eval_env, n_eval_episodes=20, return_episode_rewards=True
)
for i, (r, l) in enumerate(zip(episode_rewards, episode_lengths)):
    print(f"Episode {i+1}: reward={r:.1f}, length={l}")

print(f"\nMean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
eval_env.close()