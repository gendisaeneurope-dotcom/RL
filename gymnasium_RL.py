import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

# ── Get the base class without relying on internal module paths ──
_base_env = gym.make("InvertedPendulum-v5").unwrapped
InvertedPendulumEnv = type(_base_env)


# ── Custom environment with your own reward ──
class MyInvertedPendulumEnv(InvertedPendulumEnv):
    def step(self, action):
        self.do_simulation(action, self.frame_skip)  # apply action, advance physics
        observation = self._get_obs()  # read new state [cart_pos, pole_angle, cart_vel, pole_vel]
        terminated = bool(not np.isfinite(observation).all() or (np.abs(observation[1]) > 0.2))  # fail if angle > 0.2 rad

        com_deviation = observation[1]**2       # penalty: how far pole tilted
        control_effort = float(action[0])**2    # penalty: how hard cart was pushed
        reward = -(1.0 * com_deviation + 0.1 * control_effort) if not terminated else -10.0  # custom weighted cost

        info = {"reward_survive": reward}
        if self.render_mode == "human":
            self.render()  # only draw window if rendering requested
        return observation, reward, terminated, False, info


# ── Training ──
env = TimeLimit(MyInvertedPendulumEnv(), max_episode_steps=1000)  # cap episode at 1000 steps
model = PPO("MlpPolicy", env, verbose=1)   # create PPO agent
model.learn(total_timesteps=50_000)        # train for 50k steps
model.save("ppo_inverted_pendulum_custom") # save trained weights
env.close()


# ── Evaluation with rendering ──
model = PPO.load("ppo_inverted_pendulum_custom")  # reload saved model
env = TimeLimit(MyInvertedPendulumEnv(render_mode="human"), max_episode_steps=1000)  # env with visual window
obs, info = env.reset()  # reset to starting state

for _ in range(1000):
    action, _ = model.predict(obs)                          # ask model for action
    obs, reward, terminated, truncated, info = env.step(action)  # apply it
    if terminated or truncated:
        obs, info = env.reset()  # restart if pole fell or time limit hit
env.close()


# ── Quantitative evaluation ──
eval_env = TimeLimit(MyInvertedPendulumEnv(), max_episode_steps=1000)  # non-rendering env, faster

episode_rewards, episode_lengths = evaluate_policy(
    model, eval_env, n_eval_episodes=20, return_episode_rewards=True  # run 20 episodes, get raw lists
)
for i, (r, l) in enumerate(zip(episode_rewards, episode_lengths)):
    print(f"Episode {i+1}: reward={r:.1f}, length={l}")  # per-episode result

print(f"\nMean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")  # overall average + consistency
print(episode_rewards[:5])  # raw values, no rounding

eval_env.close()