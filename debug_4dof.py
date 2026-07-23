import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from RL_4DoF import My4DOFEnv, N_JOINTS

debug_env = My4DOFEnv()
obs, info = debug_env.reset()
log = []
for step in range(500):
    obs, reward, terminated, truncated, info = debug_env.step(np.zeros(N_JOINTS))
    log.append(np.degrees(debug_env.data.qpos[:N_JOINTS]))
    if terminated:
        break
log = np.array(log)
print("Angle drift direction (first vs last 10 steps):")
print("Start:", log[:10].mean(axis=0))
print("End:", log[-10:].mean(axis=0))
debug_env.close()