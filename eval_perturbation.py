"""Evaluate a policy (trained WITHOUT perturbation) under perturbations
applied only at test time. Compares against the same conditions with no
perturbation, to isolate the effect of the disturbance.

For target-reaching runs: sweeps the target grid.
For preliminary runs (no target, so "target" here just means holding near
center): repeats several seeds instead of a grid.

  python eval_perturbation.py runs/xcom_w1_s0
  python eval_perturbation.py runs/xcom_w1_s0 --prob 0.02 --force 30
  python eval_perturbation.py runs/preliminary_s0
"""
import argparse
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from postural_env import PosturalEnv, TARGET_RANGE, load_run_config


def rollout(model, run_dir, cfg, target, disturb_prob, force_range, seed):
    def make():
        e = PosturalEnv(mode=cfg["mode"], safety=cfg["safety"], safety_weight=0.0,
                        fixed_target=target, disturb_prob=disturb_prob,
                        force_range=force_range)
        e = TimeLimit(e, max_episode_steps=1000)
        e.reset(seed=seed)
        return e
    venv = DummyVecEnv([make])
    venv = VecNormalize.load(f"{run_dir}/vecnormalize.pkl", venv)
    venv.training = False
    venv.norm_reward = False
    obs = venv.reset()
    info = [{}]
    max_dev = 0.0
    target_for_dev = target if target is not None else 0.0
    for _ in range(1000):
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = venv.step(act)
        max_dev = max(max_dev, abs(info[0]["com_y"] - target_for_dev))
        if done[0]:
            break
    venv.close()
    return info[0]["com_y"], info[0]["failed"], max_dev


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--prob", type=float, default=0.01,
                    help="Per-step probability of a perturbation force.")
    p.add_argument("--force", type=float, default=20.0,
                    help="Max perturbation force magnitude (N).")
    p.add_argument("--n-seeds", type=int, default=3,
                    help="Repeats per condition (perturbation is random).")
    a = p.parse_args()

    cfg = load_run_config(a.run_dir)
    model = PPO.load(f"{a.run_dir}/model")

    if cfg["mode"] == "preliminary":
        conditions = list(range(11))   # 11 repeats, no target to sweep
        is_target_mode = False
        col_label = "seed"
    else:
        conditions = list(np.linspace(-TARGET_RANGE, TARGET_RANGE, 11))
        is_target_mode = True
        col_label = "target"

    print(f"Mode: {cfg['mode']} (safety={cfg['safety']})")
    print(f"Perturbation: prob={a.prob}/step, force range=+-{a.force}N, "
          f"{a.n_seeds} seeds/condition")
    print(f"\n{col_label:>8} | {'clean err(mm)':>13} {'fail':>5} | "
          f"{'perturbed err(mm)':>18} {'fail':>5} {'max dev(mm)':>12}")
    print("-" * 75)

    clean_errs, pert_errs = [], []
    clean_fails, pert_fails = 0, 0
    n = 0

    for c in conditions:
        t = c if is_target_mode else None
        clean_seed = 0 if is_target_mode else c
        c_com, c_fail, _ = rollout(model, a.run_dir, cfg, t, 0.0, (0, 0), seed=clean_seed)
        c_err = abs(c_com - (t or 0.0)) * 1000
        clean_errs.append(c_err)
        clean_fails += int(c_fail)

        p_errs, p_fails, p_devs = [], [], []
        for s in range(a.n_seeds):
            seed = (100 + s) if is_target_mode else (100 + c * 10 + s)
            p_com, p_fail, p_dev = rollout(model, a.run_dir, cfg, t, a.prob,
                                           (-a.force, a.force), seed=seed)
            p_errs.append(abs(p_com - (t or 0.0)) * 1000)
            p_fails.append(p_fail)
            p_devs.append(p_dev * 1000)
        pert_errs.extend(p_errs)
        pert_fails += sum(p_fails)
        n += a.n_seeds

        label = c if is_target_mode else float(c)
        print(f"{label:>8.4f} | {c_err:>13.2f} {str(c_fail):>5} | "
              f"{np.mean(p_errs):>18.2f} {sum(p_fails)}/{a.n_seeds:>3} "
              f"{np.mean(p_devs):>12.2f}")

    print("\n" + "=" * 75)
    print(f"CLEAN     mean|error|={np.mean(clean_errs):.2f}mm  "
          f"failures={clean_fails}/{len(conditions)}")
    print(f"PERTURBED mean|error|={np.mean(pert_errs):.2f}mm  "
          f"failures={pert_fails}/{n}")
    print("=" * 75)
    print("\nmax dev = largest momentary distance from target/center during "
          "the episode (shows the perturbation's immediate effect, separate "
          "from whether it recovered by the end).")


if __name__ == "__main__":
    main()
