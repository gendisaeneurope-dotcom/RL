"""Environment-only sanity check. No policy involved.
Run this BEFORE retraining as it separates env bugs from policy bugs.
"""
import numpy as np
from candidate2_yaxis import Candidate2Env

env = Candidate2Env()

print("=== BoS bounds ===")
print(f"base_half_length (AP, x): {env.base_half_length:.4f} m")
print(f"base_half_width  (ML, y): {env.base_half_width:.4f} m")
print(f"target_displacement:      {env.target_displacement:.4f} m")
print(f"  -> needs com_y to reach +/-{env.target_displacement/2:.4f} m "
      f"({100*(env.target_displacement/2)/env.base_half_width:.0f}% of ML bound)\n")

print("=== reset() check: start/target placement ===")
for ep in range(5):
    obs, info = env.reset(seed=ep)
    cx, cy = env._com_xy()
    print(f"ep {ep}: start com_y={cy:+.4f}  target_y={env.target_y:+.4f}  "
          f"displacement={abs(env.target_y - cy):.4f}  (start com_x={cx:+.4f})")

print("\n=== zero-action check: does a passive body survive? ===")
obs, info = env.reset(seed=0)
for t in range(300):
    obs, r, term, trunc, info = env.step(np.zeros(4))
    if term:
        print(f"FAILED at step {t}: com_y={info['com_y']:+.4f}, "
              f"xcom_y={info['xcom_y']:+.4f} (bound {env.base_half_width:.3f}), "
              f"com_x={info['com_x']:+.4f}, xcom_x={info['xcom_x']:+.4f} "
              f"(bound {env.base_half_length:.3f})")
        break
else:
    print(f"survived 300 steps: com_y={info['com_y']:+.4f}, com_x={info['com_x']:+.4f}")

print("\n=== random-action check (10 episodes) ===")
lengths = []
for ep in range(10):
    obs, info = env.reset(seed=100+ep)
    n = 0
    while n < 1000:
        obs, r, term, trunc, info = env.step(env.action_space.sample() * 0.3)
        n += 1
        if term:
            break
    lengths.append(n)
print(f"episode lengths: {lengths}")
print(f"mean: {np.mean(lengths):.0f}")