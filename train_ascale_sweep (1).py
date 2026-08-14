"""
train_ascale_sweep.py
======================
Tests the supervisor's suggestion: raise A_SCALE (the overall weight on
the combined energy+safety block, relative to tracking) to see whether
AP-axis excursion shrinks. Applies to all three candidates.

Runs SEEDS-many training runs per (candidate, a_scale) pair from the
start, since single-run sweeps were already shown unreliable (see
train_safety_sweep.py, train_offaxis_sweep_c1.py, train_safety_sweep_c3.py
seed-check results).

Baseline (a=1.0) results already confirmed at the corrected geometry:
    C1 off_axis=0.15:  20/20, 20/20, 20/20  (mean err 0.0008-0.0027)
    C2 safety=0.15:    20/20, 20/20, 20/20  (mean err 0.0005-0.0012)
                       AP p-p 0.0143, 0.0730, 0.0394
    C3 safety=0.50:    20/20, 20/20, 20/20  (mean err 0.0012-0.0022)
                       AP p-p 0.0225, 0.0570, 0.0210

Usage:
    python train_ascale_sweep.py --candidate candidate2 --a_values 1.0 2.0 4.0
    python train_ascale_sweep.py --candidate candidate2 --a_values 1.0 2.0 4.0 8.0
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import importlib
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv

from candidate1_yaxis import Candidate1Env
from candidate2_yaxis import Candidate2Env
from candidate3_yaxis import Candidate3Env

TRAIN_STEPS = 3_000_000   # matches the confirmed-stable protocol, not the noisy 1M screen
SEEDS = [0, 1, 2]
EPS_POS = 0.005

# Fixed weights per candidate, matching the seed-confirmed baseline above.
# Only A_SCALE varies in this sweep.
FIXED_CFG = {
    "candidate1": dict(env_cls=Candidate1Env, off_axis_weight=0.15),
    "candidate2": dict(env_cls=Candidate2Env, safety_weight=0.15, off_axis_weight=0.30),
    "candidate3": dict(env_cls=Candidate3Env, safety_weight=0.50, off_axis_weight=1.00),
}


def make_env_instance(cfg, a_scale, **overrides):
    """Instantiates the candidate env with a_scale applied via module
    patch, since A_SCALE is a module-level constant in each candidate file
    rather than a constructor argument. Patching module globals is
    required here -- do not skip this, changing it in only one imported
    location will silently leave other candidates' modules at a=1.0."""
    mod = importlib.import_module(cfg["env_cls"].__module__)
    mod.A_SCALE = a_scale
    kwargs = {k: v for k, v in cfg.items() if k != "env_cls"}
    kwargs.update(overrides)
    return cfg["env_cls"](**kwargs)


def train_one(candidate_name, a_scale, seed):
    cfg = FIXED_CFG[candidate_name]
    tag = f"{candidate_name}_ascale{a_scale:g}_s{seed}".replace(".", "")
    log_dir = f"./training_logs_{tag}/"
    os.makedirs(log_dir, exist_ok=True)

    def make_env(rank):
        def _f():
            e = make_env_instance(cfg, a_scale, disturb_prob=0.1, force_range=(0, 30))
            e = TimeLimit(e, max_episode_steps=1000)
            e = Monitor(e, os.path.join(log_dir, f"monitor_{rank}"))
            return e
        return _f

    N_ENVS = 8
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    env.seed(int(seed))
    env = VecNormalize(env, norm_obs=False, norm_reward=False)

    model = PPO("MlpPolicy", env, n_steps=2048, batch_size=256, ent_coef=0.01,
                learning_rate=3e-4, gamma=0.99, verbose=1, seed=int(seed))
    model.learn(total_timesteps=TRAIN_STEPS)
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
    p.add_argument("--a_values", type=float, nargs="+", required=True,
                   help="A_SCALE values to test, e.g. 1.0 2.0 4.0")
    args = p.parse_args()

    results = []
    for a in args.a_values:
        for seed in SEEDS:
            print(f"\n=== {args.candidate}, A_SCALE={a}, seed={seed} ===")
            tag = train_one(args.candidate, a, seed)
            err, ap, wrong, hit = evaluate(tag, args.candidate, a)
            results.append((a, seed, err, ap, wrong, hit))
            print(f"  err {err:.4f} | AP p-p {ap:.4f} | wrong {wrong}/20 | hit {hit}/20")

    print("\n" + "=" * 80)
    print(f"A_SCALE SWEEP -- {args.candidate}")
    print(f"{'a':>6} {'seed':>5} {'mean_err':>10} {'AP p-p':>9} {'wrong':>7} {'hit':>6}")
    print("-" * 80)
    for a, seed, err, ap, wrong, hit in results:
        print(f"{a:>6.1f} {seed:>5} {err:>10.4f} {ap:>9.4f} {wrong:>5}/20 {hit:>4}/20")
    print("=" * 80)

    print("\nPer-a_scale mean AP p-p (averaged across seeds):")
    for a in args.a_values:
        aps = [r[3] for r in results if r[0] == a]
        print(f"  a={a:g}: mean AP p-p = {np.mean(aps):.4f}  (individual: {[f'{x:.4f}' for x in aps]})")
    print("\nCompare against a=1.0 baseline AP p-p reported in the docstring above.")
    print("If AP p-p decreases meaningfully and monotonically as 'a' increases,")
    print("that supports the supervisor's hypothesis. If it doesn't move, or")
    print("moves inconsistently across seeds, report that directly -- don't")
    print("infer a trend from fewer than 3 seeds per point.")
