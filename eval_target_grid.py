import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO

from RL_4DoF_reward_candidate1 import My4DOFTargetEnv, TARGET_RANGE

model = PPO.load("ppo_ankle_hip_reward_candidate2_xcom")

# Evenly spaced grid across the full trained target range, plus a couple of
# points just outside it to see how the policy handles slight extrapolation.
# Fixed, not random -- gives a defensible precision-vs-target-magnitude table.
grid = np.linspace(-TARGET_RANGE, TARGET_RANGE, 11)
extrapolation_points = [-TARGET_RANGE * 1.2, TARGET_RANGE * 1.2]
targets = list(grid) + extrapolation_points

results = []
for target in targets:
    env = TimeLimit(My4DOFTargetEnv(disturb_prob=0.0, fixed_target=target), max_episode_steps=1000)
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()

    error = info["com_y"] - info["target_y"]
    results.append((target, info["com_y"], error, info["h"], truncated))

print(f"{'target_y':>10} | {'final_com_y':>12} | {'error':>9} | {'h':>7} | reached_full_length")
print("-" * 65)
for target, com_y, error, h, full_length in results:
    tag = " (extrapolation)" if abs(target) > TARGET_RANGE else ""
    print(f"{target:>10.4f} | {com_y:>12.4f} | {error:>9.4f} | {h:>7.4f} | {full_length}{tag}")

errors = np.array([r[2] for r in results])
in_range_errors = np.array([r[2] for r in results if abs(r[0]) <= TARGET_RANGE])
print(f"\nMean |error| (in-range targets only): {np.mean(np.abs(in_range_errors)):.4f}")
print(f"Max |error| (in-range targets only): {np.max(np.abs(in_range_errors)):.4f}")
print(f"Mean |error| (all, incl. extrapolation): {np.mean(np.abs(errors)):.4f}")
