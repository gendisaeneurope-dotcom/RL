"""Compare a trained policy's behavior to the real subject's trials at a
matched target.

  python compare_to_real.py runs/xcom_w1_s0 real_trials_summary.csv
"""
import sys
import numpy as np
import pandas as pd
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from postural_env import PosturalEnv

run_dir = sys.argv[1]
summary_path = sys.argv[2]

real = pd.read_csv(summary_path)
target_m = real["com0_final"].abs().mean() / 1000.0  # real target, meters
print(f"Real subject's average target magnitude: {target_m*1000:.1f} mm "
      f"({target_m:.4f} m)")

for real_side, sign in [("right", +1), ("left", -1)]:
    sub = real[real.side == real_side]
    real_final_mm = sub.com0_final.mean()
    real_final_std = sub.com0_final.std()

    t = sign * target_m
    model = PPO.load(f"{run_dir}/model")
    def make_env(t=t):
        e = PosturalEnv(safety="none", safety_weight=0.0, fixed_target=t)
        e = TimeLimit(e, max_episode_steps=1000)
        e.reset(seed=0)
        return e
    venv = DummyVecEnv([make_env])
    venv = VecNormalize.load(f"{run_dir}/vecnormalize.pkl", venv)
    venv.training = False
    venv.norm_reward = False
    obs = venv.reset()
    info = [{}]
    for _ in range(1000):
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = venv.step(act)
        if done[0]:
            break
    sim_final_mm = info[0]["com_y"] * 1000
    venv.close()

    print(f"\n{real_side.upper()} target ({t*1000:+.1f} mm):")
    print(f"  real subject : {real_final_mm:+7.1f} mm (n={len(sub)}, std={real_final_std:.1f})")
    print(f"  sim policy   : {sim_final_mm:+7.1f} mm  failed={info[0]['failed']}")
    print(f"  difference   : {abs(real_final_mm - sim_final_mm):.1f} mm")

print("\nNote: this compares the FINAL held position only. If you want full")
print("trajectory shape (not just endpoint), tell me and I'll extend this")
print("to plot both time series together -- needs one real trial's raw")
print("timesteps (from real_trials_extracted.csv) aligned to the sim's.")
