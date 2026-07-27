import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from RL_4DoF_target import My4DOFTargetEnv

# Loads the model trained WITHOUT any perturbation (per supervisor's request).
# This script is where perturbation is introduced for the first time -- purely
# at test time, to check whether the policy generalizes to disturbance it
# never saw during training.
model = PPO.load("ppo_ankle_hip_target")

# Escalating conditions. Report all of them, not just the best-looking one --
# the point is to see where (if anywhere) target-reaching starts to break.
conditions = [
    {"disturb_prob": 0.0, "force_range": (-20, 20)},     # sanity check / baseline (no disturbance)
    {"disturb_prob": 0.05, "force_range": (-20, 20)},    # light, occasional
    {"disturb_prob": 0.3, "force_range": (-20, 20)},     # moderate, same force as before
    {"disturb_prob": 1.0, "force_range": (-20, 20)},     # constant, same force
    {"disturb_prob": 1.0, "force_range": (-100, 100)},   # constant, much stronger
]

for cond in conditions:
    print(f"\n=== disturb_prob={cond['disturb_prob']}, force_range={cond['force_range']} ===")

    # Reward/length summary
    eval_env = TimeLimit(My4DOFTargetEnv(**cond), max_episode_steps=1000)
    eval_env = Monitor(eval_env)
    rewards, lengths = evaluate_policy(model, eval_env, n_eval_episodes=20, return_episode_rewards=True)
    eval_env.close()
    print(f"Mean reward: {np.mean(rewards):.4f} +/- {np.std(rewards):.4f}  |  "
          f"episodes at full length: {sum(1 for l in lengths if l == 1000)}/20")

    # Does it still reach the target despite the disturbance? This is the
    # specific question the supervisor asked -- reward/length alone can't
    # answer it, need final com_y vs target_y directly.
    check_env = TimeLimit(My4DOFTargetEnv(**cond), max_episode_steps=1000)
    errors = []
    for ep in range(10):
        obs, info = check_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = check_env.step(action)
            done = terminated or truncated
        err = info["com_y"] - info["target_y"]
        errors.append(err)
        print(f"  target_y={info['target_y']:.4f}, final com_y={info['com_y']:.4f}, error={err:.4f}")
    check_env.close()
    print(f"Mean |error| across 10 episodes: {np.mean(np.abs(errors)):.4f}")
