import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from gymnasium import spaces
import numpy as np


# ── Load custom 4-DOF ankle-hip model via xml_file ──
xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ankle_knee_hip_trunk.xml")
_base_env = gym.make("InvertedPendulum-v5", xml_file=xml_path).unwrapped
AnkleKneeHipTrunkEnv = type(_base_env)

JOINT_NAMES = ["ankle", "knee", "hip", "trunk"]
N_JOINTS = 4
ANGLE_LIMIT = 0.4  # rad, matches trunk's tighter joint range (most restrictive)

class My4DOFEnv(AnkleKneeHipTrunkEnv):
    def __init__(self, disturb_prob=0.0, force_range=(-20, 20), omega=0.1, **kwargs):
        super().__init__(xml_file=xml_path, **kwargs)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(N_JOINTS,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 * N_JOINTS,), dtype=np.float64)
        self._current_step = 0
        self._max_steps = 1000
        self.disturb_prob = disturb_prob
        self.force_range = force_range
        self.omega = omega
        self.trunk_body_id = self.model.body("trunk").id

    def reset(self, **kwargs):
        self._current_step = 0
        return super().reset(**kwargs)

    def _get_obs(self):
        angles = self.data.qpos[:N_JOINTS].copy()
        angvels = self.data.qvel[:N_JOINTS].copy()
        return np.concatenate([angles, angvels]).astype(np.float64)

    def step(self, action):
        if self.disturb_prob > 0 and np.random.rand() < self.disturb_prob:
            force = np.random.uniform(*self.force_range)
            # apply to trunk body (last body in the chain)
            self.data.xfrc_applied[self.trunk_body_id, 0] = force
        else:
            self.data.xfrc_applied[self.trunk_body_id, 0] = 0.0

        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        angles = observation[:N_JOINTS]

        failed = bool(
            not np.isfinite(observation).all()
            or np.any(np.abs(angles) > ANGLE_LIMIT)
        )

        h = np.mean(np.cos(angles))  # average upright measure across all 4 joints

        self._current_step += 1
        success = (self._current_step >= self._max_steps) and not failed

        if success:
            reward = 1000.0
            terminated = True
        elif failed:
            reward = -100.0 - 400.0 * (1.0 - h)
            terminated = True
        else:
            effort = np.sum(np.square(action))
            reward = h - self.omega * effort
            terminated = False

        info = {"reward_survive": reward}
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info


# ── Training ──
log_dir = "./training_logs_4dof/"
os.makedirs(log_dir, exist_ok=True)

env = TimeLimit(My4DOFEnv(), max_episode_steps=1000)
env = Monitor(env, log_dir)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb_logs_4dof/")
model.learn(total_timesteps=500_000)
model.save("ppo_ankle_knee_hip_trunk")
env.close()


# ── Evaluation with rendering (quick visual check) ──
model = PPO.load("ppo_ankle_knee_hip_trunk")
env = TimeLimit(My4DOFEnv(render_mode="human"), max_episode_steps=1000)
obs, info = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()


# ── Quantitative evaluation ──
eval_env = TimeLimit(My4DOFEnv(), max_episode_steps=1000)

episode_rewards, episode_lengths = evaluate_policy(
    model, eval_env, n_eval_episodes=20, return_episode_rewards=True
)
for i, (r, l) in enumerate(zip(episode_rewards, episode_lengths)):
    print(f"Episode {i+1}: reward={r:.1f}, length={l}")

print(f"\nMean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
eval_env.close()