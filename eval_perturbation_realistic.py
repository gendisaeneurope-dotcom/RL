"""Test a policy (trained WITHOUT perturbation, per supervisor's instruction)
against the REAL experimental perturbation: continuous, anterior, scaled to
the subject's own CoM velocity -- not the old random-push mechanism.

  python eval_perturbation_realistic.py runs/capture_w1_s0
  python eval_perturbation_realistic.py runs/capture_w1_s0 --gain 100
"""
import argparse
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from postural_env import PosturalEnv, TARGET_RANGE, load_run_config


def rollout(run_dir, cfg, target, gain, max_force, seed):
    def make():
        e = PosturalEnv(mode=cfg["mode"], safety=cfg["safety"], safety_weight=0.0,
                        fixed_target=target, perturb_vel_gain=gain,
                        perturb_max_force=max_force)
        e = TimeLimit(e, max_episode_steps=1000)
        e.reset(seed=seed)
        return e
    venv = DummyVecEnv([make])
    venv = VecNormalize.load(f"{run_dir}/vecnormalize.pkl", venv)
    venv.training = False
    venv.norm_reward = False
    obs = venv.reset()
    info = [{}]
    max_com_x_dev = 0.0
    for _ in range(1000):
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = venv.step(act)
        max_com_x_dev = max(max_com_x_dev, abs(info[0]["com_x"]))
        if done[0]:
            break
    venv.close()
    return info[0], max_com_x_dev


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--gain", type=float, default=50.0,
                    help="N per m/s of CoM anterior velocity (real protocol: "
                         "continuous, scaled to subject's own velocity).")
    p.add_argument("--max-force", type=float, default=100.0)
    a = p.parse_args()

    cfg = load_run_config(a.run_dir)
    global model
    model = PPO.load(f"{a.run_dir}/model")

    if cfg["mode"] == "preliminary":
        targets = [None] * 5
    else:
        targets = list(np.linspace(-TARGET_RANGE, TARGET_RANGE, 11))

    print(f"Mode: {cfg['mode']} (safety={cfg['safety']})")
    print(f"Perturbation: continuous, gain={a.gain} N/(m/s), "
          f"capped at +-{a.max_force}N\n")

    print(f"{'target':>8} | {'clean err(mm)':>13} {'fail':>5} | "
          f"{'perturbed err(mm)':>18} {'fail':>5} {'max |com_x|(mm)':>16}")
    print("-" * 80)

    for i, t in enumerate(targets):
        clean_info, _ = rollout(a.run_dir, cfg, t, 0.0, 0.0, seed=i)
        pert_info, max_x = rollout(a.run_dir, cfg, t, a.gain, a.max_force, seed=i)

        t_val = t if t is not None else 0.0
        c_err = abs(clean_info["com_y"] - t_val) * 1000
        p_err = abs(pert_info["com_y"] - t_val) * 1000

        label = t_val if t is not None else i
        print(f"{label:>8.4f} | {c_err:>13.2f} {str(clean_info['failed']):>5} | "
              f"{p_err:>18.2f} {str(pert_info['failed']):>5} {max_x*1000:>16.2f}")


if __name__ == "__main__":
    main()
