"""
train_omega_sweep.py
======================
push omega toward 0 so the
safety proportion (1-omega) dominates the energy proportion (omega)
INSIDE the energy/safety split, as opposed to the earlier A_SCALE sweep
(train_ascale_sweep.py) which scaled the whole energy+safety block
uniformly without changing this internal ratio at all. These are two
different levers in the same formula:

    R = tracking + shaping + a*(omega*energy + (1-omega)*safety) - off_axis

A_SCALE sweep varied `a` with omega fixed at 0.2.
THIS sweep varies `omega` with `a` fixed at 1.0 (the reported-results value).


!!!!!!!!!!!!!!!!!!! omega -> 0 means the energy !!!!!!!!!!!!!!!!!!!!
penalty on actuator torque effort nearly vanishes from the reward. This
may reduce AP excursion as hypothesized, but likely at the cost of much
larger, less efficient joint torques, since almost nothing discourages
them anymore. Report torque magnitudes alongside AP excursion, not AP
excursion alone, when interpreting this sweep.

Given the seed-variance problems already documented twice in this project
(safety/off-axis weight sweeps, and the A_SCALE sweep), this sweep uses
the same 3-seeds-per-value discipline as both of those.

Usage (run ONE combination per invocation, same pattern as train_ascale_sweep.py):
    python train_omega_sweep.py --candidate candidate2 --omega_value 0.2 --seed 0
    python train_omega_sweep.py --candidate candidate2 --omega_value 0.05 --seed 0
    python train_omega_sweep.py --candidate candidate2 --omega_value 0.01 --seed 0

Poweshell run:
    foreach ($w in 0.2,0.05,0.01) {
      foreach ($s in 0,1,2) {
        python train_omega_sweep.py --candidate candidate2 --omega_value $w --seed $s
      }
    }
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import importlib
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv

from candidate1_yaxis import Candidate1Env
from candidate2_yaxis import Candidate2Env
from candidate3_yaxis import Candidate3Env

TRAIN_STEPS = 3_000_000
CHECKPOINT_FREQ = 250_000
EPS_POS = 0.005
FIXED_A_SCALE = 1.0   # held fixed -- this sweep isolates omega only

FIXED_CFG = {
    "candidate1": dict(env_cls=Candidate1Env, off_axis_weight=0.15),
    "candidate2": dict(env_cls=Candidate2Env, safety_weight=0.15, off_axis_weight=0.30),
    "candidate3": dict(env_cls=Candidate3Env, safety_weight=0.50, off_axis_weight=1.00),
}


def make_env_instance(cfg, omega_value, **overrides):
    mod = importlib.import_module(cfg["env_cls"].__module__)
    mod.A_SCALE = FIXED_A_SCALE
    kwargs = {k: v for k, v in cfg.items() if k != "env_cls"}
    kwargs["omega"] = omega_value
    kwargs.update(overrides)
    return cfg["env_cls"](**kwargs)


def train_one(candidate_name, omega_value, seed):
    cfg = FIXED_CFG[candidate_name]
    tag = f"{candidate_name}_omega{omega_value:g}_s{seed}".replace(".", "")
    log_dir = f"./training_logs_{tag}/"
    ckpt_dir = f"./checkpoints_{tag}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    existing = sorted(
        [f for f in os.listdir(ckpt_dir) if f.endswith(".zip")],
        key=lambda f: int(f.split("_")[-2]) if f.split("_")[-2].isdigit() else 0
    )

    def make_env(rank):
        def _f():
            e = make_env_instance(cfg, omega_value, disturb_prob=0.1, force_range=(0, 30))
            e = TimeLimit(e, max_episode_steps=1000)
            e = Monitor(e, os.path.join(log_dir, f"monitor_{rank}"))
            return e
        return _f

    N_ENVS = 4
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    env.seed(int(seed))
    env = VecNormalize(env, norm_obs=False, norm_reward=False)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path=ckpt_dir,
        name_prefix="ckpt",
        save_vecnormalize=True,
    )

    if existing:
        latest = os.path.join(ckpt_dir, existing[-1])
        print(f"Auto-resuming from checkpoint: {latest} (no action needed)")
        model = PPO.load(latest, env=env)
        vecnorm_ckpt = latest.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(vecnorm_ckpt):
            env = VecNormalize.load(vecnorm_ckpt, env.venv)
            model.set_env(env)
        remaining = TRAIN_STEPS - model.num_timesteps
        if remaining > 0:
            model.learn(total_timesteps=remaining, callback=checkpoint_cb, reset_num_timesteps=False)
    else:
        model = PPO("MlpPolicy", env, n_steps=2048, batch_size=256, ent_coef=0.01,
                    learning_rate=3e-4, gamma=0.99, verbose=0, seed=int(seed))
        model.learn(total_timesteps=TRAIN_STEPS, callback=checkpoint_cb)

    model.save(f"ppo_{tag}")
    env.save(f"vecnormalize_{tag}.pkl")
    env.close()
    return tag


def evaluate(tag, candidate_name, omega_value, n_eps=20):
    cfg = FIXED_CFG[candidate_name]
    model = PPO.load(f"ppo_{tag}")
    errors, ap_ptp, torque_rms, wrong_side = [], [], [], 0
    for ep in range(n_eps):
        def _f():
            e = make_env_instance(cfg, omega_value)
            return TimeLimit(e, max_episode_steps=1000)
        venv = DummyVecEnv([_f])
        venv = VecNormalize.load(f"vecnormalize_{tag}.pkl", venv)
        venv.training = False
        venv.norm_reward = False
        venv.seed(ep)
        obs = venv.reset()
        done, ap_trace, torque_trace = False, [], []
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, d, info = venv.step(a)
            done = bool(d[0])
            ap_trace.append(float(info[0]["com_x"]))
            torque_trace.append(float(np.mean(np.square(a))))
        i = info[0]
        errors.append(abs(i["com_y"] - i["target_y"]))
        ap_ptp.append(float(np.ptp(ap_trace)))
        torque_rms.append(float(np.sqrt(np.mean(torque_trace))))
        if np.sign(i["com_y"]) != np.sign(i["target_y"]):
            wrong_side += 1
        venv.close()
    return (float(np.mean(errors)), float(np.mean(ap_ptp)), float(np.mean(torque_rms)),
            wrong_side, sum(1 for e in errors if e < EPS_POS))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True, choices=list(FIXED_CFG.keys()))
    p.add_argument("--omega_value", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()

    tag = train_one(args.candidate, args.omega_value, args.seed)
    err, ap, torque, wrong, hit = evaluate(tag, args.candidate, args.omega_value)

    print(f"RESULT  omega={args.omega_value} seed={args.seed}: "
          f"err={err:.4f}  AP_p-p={ap:.4f}  torque_rms={torque:.4f}  wrong={wrong}/20  hit={hit}/20")
