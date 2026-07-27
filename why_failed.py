"""Why did it fail? Reports WHICH joint hit its limit, WHEN, and what the
sagittal (front-back) joints were doing at the time.

The reward only cares about mediolateral CoM. The two sagittal joints
(ankle_flexion, hip_flexion) have no target and no penalty beyond effort
cost, so they are a prime suspect for slow uncontrolled drift.

For target-reaching runs: sweeps the target grid.
For preliminary runs (no target): repeats several seeds instead.

  python why_failed.py runs/none_w0_s0
  python why_failed.py runs/preliminary_s0
"""
import sys
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from postural_env import PosturalEnv, TARGET_RANGE, JOINT_RANGE, JOINT_NAMES, FAIL_MARGIN, load_run_config

LIMIT = JOINT_RANGE * FAIL_MARGIN


def _rollout(run_dir, model, cfg, fixed_target, seed):
    def make():
        e = PosturalEnv(mode=cfg["mode"], safety=cfg["safety"], safety_weight=0.0,
                        fixed_target=fixed_target)
        e = TimeLimit(e, max_episode_steps=1000)
        e.reset(seed=seed)
        return e
    venv = DummyVecEnv([make])
    venv = VecNormalize.load(f"{run_dir}/vecnormalize.pkl", venv)
    venv.training = False
    venv.norm_reward = False
    obs = venv.reset()
    raw = venv.venv.envs[0].unwrapped
    n = 0
    q = raw.data.qpos[:4].copy()
    info = [{}]
    for _ in range(1000):
        act, _ = model.predict(obs, deterministic=True)
        q = raw.data.qpos[:4].copy()
        obs, _, done, info = venv.step(act)
        n += 1
        if done[0]:
            break
    venv.close()
    return n, q, info[0]


def run(run_dir):
    cfg = load_run_config(run_dir)
    model = PPO.load(f"{run_dir}/model")

    if cfg["mode"] == "preliminary":
        conditions = [("seed", s) for s in range(11)]
        header = f"{'seed':>8}"
    else:
        conditions = [("target", t) for t in np.linspace(-TARGET_RANGE, TARGET_RANGE, 11)]
        header = f"{'target':>8}"

    print(f"{header} {'steps':>6} {'outcome':>8} {'culprit joint':>16} "
          f"{'|q|/limit at end (ev, aflex, abd, hflex)'}")
    print("-" * 100)

    for kind, val in conditions:
        if kind == "target":
            n, q, info = _rollout(run_dir, model, cfg, fixed_target=val, seed=0)
        else:
            n, q, info = _rollout(run_dir, model, cfg, fixed_target=None, seed=val)

        frac = np.abs(q) / LIMIT
        worst = int(np.argmax(frac))
        failed = info.get("failed", False)
        culprit = JOINT_NAMES[worst] if failed else "-"
        print(f"{val:>8.4f} {n:>6d} {'FELL' if failed else 'ok':>8} "
              f"{culprit:>16} " + "  ".join(f"{f:.2f}" for f in frac))

    print("\nRead the last four columns as 'fraction of the joint's limit'.")
    print("1.00 means that joint is at the limit that ends the episode.")
    print("ev/abd move the body SIDEWAYS (what the reward is about).")
    print("aflex/hflex move it FRONT-BACK (nothing in the reward controls these).")


if __name__ == "__main__":
    run(sys.argv[1])