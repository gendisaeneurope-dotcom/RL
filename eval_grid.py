"""Fixed-grid target evaluation. Loads the matching VecNormalize statistics --
omitting these silently breaks evaluation once norm_obs=True.

Only meaningful for target-reaching runs (candidates 1/2/3). Preliminary has
no target, so there's nothing to grid-test -- use eval_perturbation.py for
that instead.

  python eval_grid.py runs/xcom_w1_s0
"""
import sys, numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from postural_env import PosturalEnv, TARGET_RANGE, load_run_config


def run(run_dir, n_grid=11):
    cfg = load_run_config(run_dir)
    if cfg["mode"] == "preliminary":
        print(f"{run_dir} was trained in 'preliminary' mode (no target). "
              f"eval_grid.py doesn't apply here -- use eval_perturbation.py "
              f"instead, which tests stability with/without a push.")
        return

    model = PPO.load(f"{run_dir}/model")
    targets = list(np.linspace(-TARGET_RANGE, TARGET_RANGE, n_grid)) \
              + [-TARGET_RANGE * 1.2, TARGET_RANGE * 1.2]
    rows = []
    for t in targets:
        venv = DummyVecEnv([lambda t=t: TimeLimit(
            PosturalEnv(mode=cfg["mode"], safety=cfg["safety"], safety_weight=0.0,
                       fixed_target=t, use_shaping=cfg.get("use_shaping", False)),
            max_episode_steps=1000)])
        venv = VecNormalize.load(f"{run_dir}/vecnormalize.pkl", venv)
        venv.training = False; venv.norm_reward = False
        obs = venv.reset()
        info = [{}]
        for _ in range(1000):
            act, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = venv.step(act)
            if done[0]:
                break
        i = info[0].get("terminal_observation") is not None and info[0] or info[0]
        rows.append((t, i["com_y"], i["com_y"] - i["target_y"], i["failed"]))
        venv.close()

    print(f"{'target':>9} {'final com_y':>12} {'error(mm)':>10} {'failed':>7}")
    for t, c, e, f in rows:
        tag = "  (extrap)" if abs(t) > TARGET_RANGE else ""
        print(f"{t:>9.4f} {c:>12.4f} {1000*e:>10.2f} {str(f):>7}{tag}")

    inr = np.array([abs(r[2]) for r in rows if abs(r[0]) <= TARGET_RANGE])
    coms = np.array([r[1] for r in rows if abs(r[0]) <= TARGET_RANGE])
    tgts = np.array([r[0] for r in rows if abs(r[0]) <= TARGET_RANGE])
    print(f"\nmean |error| = {1000*inr.mean():.2f} mm   max = {1000*inr.max():.2f} mm")
    print(f"corr(target, com_y) = {np.corrcoef(tgts, coms)[0,1]:.4f}   "
          f"(a fixed-point policy scores near 0)")
    print(f"failures: {sum(r[3] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    run(sys.argv[1])