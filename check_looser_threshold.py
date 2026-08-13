import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from train_safety_sweep_c2 import Candidate2EnvY

EPS_POS_LOOSE = 0.02   # was 0.005 -- testing whether the threshold, not the policy, was the problem

def evaluate(tag, safety_weight, n_eps=20):
    model = PPO.load(f"ppo_{tag}")
    errors, hits = [], 0
    for ep in range(n_eps):
        venv = DummyVecEnv([lambda: TimeLimit(
            Candidate2EnvY(safety_weight=safety_weight), max_episode_steps=1000)])
        venv = VecNormalize.load(f"vecnormalize_{tag}.pkl", venv)
        venv.training = False
        venv.norm_reward = False
        venv.seed(ep)
        obs = venv.reset()
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, d, info = venv.step(a)
            done = bool(d[0])
        i = info[0]
        err = abs(i["com_y"] - i["target_y"])
        errors.append(err)
        if err < EPS_POS_LOOSE:
            hits += 1
        print(f"  ep{ep}: target={i['target_y']:+.4f} final={i['com_y']:+.4f} err={err:.4f}")
        venv.close()
    return float(np.mean(errors)), hits

for seed in [0, 1, 2]:
    tag = f"c2_yaxis_sw015_s{seed}"
    err, hits = evaluate(tag, 0.15)
    print(f"seed {seed}: mean err {err:.4f} | hit (< {EPS_POS_LOOSE}) {hits}/20")