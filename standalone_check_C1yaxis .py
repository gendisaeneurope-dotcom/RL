"""
Candidate 1 -> axis-swapped, no safety term (baseline).



Usage:
    python standalone_check_C1yaxis .py
"""


import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from candidate1_yaxis import Candidate1Env

model = PPO.load("ppo_candidate1_yaxis_015")
for ep in range(5):
    venv = DummyVecEnv([lambda: TimeLimit(Candidate1Env(), max_episode_steps=1000)])
    venv = VecNormalize.load("vecnormalize_candidate1_yaxis_015.pkl", venv)
    venv.training = False; venv.norm_reward = False
    venv.seed(ep); obs = venv.reset()
    done = False; n = 0
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, d, info = venv.step(a); done = bool(d[0]); n += 1
        if n % 100 == 0:
            print(f"  step {n}: com_y={info[0]['com_y']:+.4f}")
    i = info[0]
    print(f"ep {ep}: {n} steps, com_y={i['com_y']:.4f}, "
          f"target_y={i['target_y']:.4f}, err={abs(i['com_y']-i['target_y']):.4f}")
    venv.close()