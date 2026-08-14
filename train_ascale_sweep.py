"""
train_ascale_sweep.py
======================
FIXED after a WinError 1450 crash lost an 83%-complete 3M-step run with
no checkpoint saved. Two changes:

  1. ONE (candidate, a_value, seed) COMBINATION PER INVOCATION. The
     previous version looped over all combinations inside one long-running
     Python process, accumulating subprocess/handle pressure on Windows
     across SubprocVecEnv open/close cycles until the OS refused to
     allocate more (WinError 1450). Running one combination per process
     guarantees full OS resource release between runs -- drive it
     externally, see bottom of this docstring.

  2. CHECKPOINTING. model.save() previously only happened after learn()
     fully completed, so any crash mid-run lost 100% of that run's
     progress. Now saves every CHECKPOINT_FREQ steps via SB3's
     CheckpointCallback, so a crash loses at most CHECKPOINT_FREQ steps,
     and a crashed run resumes automatically rather than restarting.

Usage (run ONE combination per invocation):
    python train_ascale_sweep.py --candidate candidate2 --a_value 1.0 --seed 0
    python train_ascale_sweep.py --candidate candidate2 --a_value 2.0 --seed 0

Drive multiple combinations from PowerShell, NOT from inside this script:
    foreach ($a in 1.0,2.0,4.0) {
      foreach ($s in 0,1,2) {
        python train_ascale_sweep.py --candidate candidate2 --a_value $a --seed $s
      }
    }
Each iteration is a fresh process; Windows fully reclaims handles between
runs this way.

To re-evaluate an already-trained model without retraining:
    python train_ascale_sweep.py --candidate candidate2 --a_value 1.0 --seed 0 --eval_only
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
CHECKPOINT_FREQ = 250_000   # save every 250k steps -- max loss on crash
EPS_POS = 0.005

FIXED_CFG = {
    "candidate1": dict(env_cls=Candidate1Env, off_axis_weight=0.15),
    "candidate2": dict(env_cls=Candidate2Env, safety_weight=0.15, off_axis_weight=0.30),
    "candidate3": dict(env_cls=Candidate3Env, safety_weight=0.50, off_axis_weight=1.00),
}


def make_env_instance(cfg, a_scale, **overrides):
    mod = importlib.import_module(cfg["env_cls"].__module__)
    mod.A_SCALE = a_scale
    kwargs = {k: v for k, v in cfg.items() if k != "env_cls"}
    kwargs.update(overrides)
    return cfg["env_cls"](**kwargs)


def train_one(candidate_name, a_scale, seed):
    cfg = FIXED_CFG[candidate_name]
    tag = f"{candidate_name}_ascale{a_scale:g}_s{seed}".replace(".", "")
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
            e = make_env_instance(cfg, a_scale, disturb_prob=0.1, force_range=(0, 30))
            e = TimeLimit(e, max_episode_steps=1000)
            e = Monitor(e, os.path.join(log_dir, f"monitor_{rank}"))
            return e
        return _f

    N_ENVS = 4   # reduced from 8 -- fewer subprocess handles per run
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
        print(f"Resuming from checkpoint: {latest}")
        model = PPO.load(latest, env=env)
        vecnorm_ckpt = latest.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(vecnorm_ckpt):
            env = VecNormalize.load(vecnorm_ckpt, env.venv)
            model.set_env(env)
        remaining = TRAIN_STEPS - model.num_timesteps
        if remaining <= 0:
            print("Checkpoint already at or past TRAIN_STEPS, skipping training.")
        else:
            print(f"Resuming: {model.num_timesteps} steps done, {remaining} remaining.")
            model.learn(total_timesteps=remaining, callback=checkpoint_cb, reset_num_timesteps=False)
    else:
        model = PPO("MlpPolicy", env, n_steps=2048, batch_size=256, ent_coef=0.01,
                    learning_rate=3e-4, gamma=0.99, verbose=1, seed=int(seed))
        model.learn(total_timesteps=TRAIN_STEPS, callback=checkpoint_cb)

    model.save(f"ppo_{tag}")
    env.save(f"vecnormalize_{tag}.pkl")
    env.close()
    return tag


def evaluate(tag, candidate_name, a_scale, n_eps=20):
    cfg = FIXED_CFG[candidate_name]
    model = PPO.load(f"ppo_{tag}")
    errors, ap_ptp, wrong_side = [], [], 0
    for ep in range(n_eps):
        def _f():
            e = make_env_instance(cfg, a_scale)
            return TimeLimit(e, max_episode_steps=1000)
        venv = DummyVecEnv([_f])
        venv = VecNormalize.load(f"vecnormalize_{tag}.pkl", venv)
        venv.training = False
        venv.norm_reward = False
        venv.seed(ep)
        obs = venv.reset()
        done, ap_trace = False, []
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, d, info = venv.step(a)
            done = bool(d[0])
            ap_trace.append(float(info[0]["com_x"]))
        i = info[0]
        errors.append(abs(i["com_y"] - i["target_y"]))
        ap_ptp.append(float(np.ptp(ap_trace)))
        if np.sign(i["com_y"]) != np.sign(i["target_y"]):
            wrong_side += 1
        venv.close()
    return (float(np.mean(errors)), float(np.mean(ap_ptp)), wrong_side,
            sum(1 for e in errors if e < EPS_POS))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True, choices=list(FIXED_CFG.keys()))
    p.add_argument("--a_value", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training, only evaluate an already-saved model.")
    args = p.parse_args()

    tag = f"{args.candidate}_ascale{args.a_value:g}_s{args.seed}".replace(".", "")

    if not args.eval_only:
        print(f"\n=== {args.candidate}, A_SCALE={args.a_value}, seed={args.seed} ===")
        tag = train_one(args.candidate, args.a_value, args.seed)

    err, ap, wrong, hit = evaluate(tag, args.candidate, args.a_value)
    print(f"\nRESULT  a={args.a_value} seed={args.seed}: "
          f"err={err:.4f}  AP_p-p={ap:.4f}  wrong={wrong}/20  hit={hit}/20")
    print(f"\nRecord this line manually -- each invocation runs one")
    print(f"combination only, no automatic aggregation across runs.")