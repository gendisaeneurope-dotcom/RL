import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor          
import numpy as np

_base_env = gym.make("InvertedPendulum-v5").unwrapped
InvertedPendulumEnv = type(_base_env)

class MyInvertedPendulumEnv(InvertedPendulumEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._current_step = 0
        self._max_steps = 1000

    def reset(self, **kwargs):
        self._current_step = 0
        return super().reset(**kwargs)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()  # [cart_pos, pole_angle, cart_vel, pole_vel]

        pole_angle = observation[1]
        cart_pos = observation[0]

        failed = bool(not np.isfinite(observation).all() or (np.abs(pole_angle) > 0.2))

        h = np.cos(pole_angle)  # 1 = fully upright, 0 = horizontal

        self._current_step += 1
        success = (self._current_step >= self._max_steps) and not failed

        if success:
            reward = 1000.0
            terminated = True
        elif failed:
            reward = -100.0 - 400.0 * (1.0 - h)
            terminated = True
        else:
            effort = float(action[0])**2
            com_x_offset = cart_pos  # cart position as proxy for horizontal offset
            omega = 0.5
            EC = omega * effort + (1 - omega) * (-com_x_offset)
            reward = h - EC
            terminated = False

        info = {"reward_survive": reward}
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

# ── Training ──
log_dir = "./training_logs/"                                   
os.makedirs(log_dir, exist_ok=True)                            

env = TimeLimit(MyInvertedPendulumEnv(), max_episode_steps=1000)
env = Monitor(env, log_dir)                                     # ADD — wraps env, logs every episode

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb_logs/")  # ADD tensorboard_log
model.learn(total_timesteps=200_000)
model.save("ppo_inverted_pendulum_custom")
env.close()

# ── Evaluation with rendering (unchanged) ──
model = PPO.load("ppo_inverted_pendulum_custom")
env = TimeLimit(MyInvertedPendulumEnv(render_mode="human"), max_episode_steps=1000)
obs, info = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()

# ── Quantitative evaluation (unchanged) ──
eval_env = TimeLimit(MyInvertedPendulumEnv(), max_episode_steps=1000)

episode_rewards, episode_lengths = evaluate_policy(
    model, eval_env, n_eval_episodes=20, return_episode_rewards=True
)
for i, (r, l) in enumerate(zip(episode_rewards, episode_lengths)):
    print(f"Episode {i+1}: reward={r:.1f}, length={l}")

print(f"\nMean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
print(episode_rewards[:5])

eval_env.close()