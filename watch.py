"""Visualize a trained policy. Opens a MuJoCo viewer window.
Works for preliminary and all target-reaching candidates (mode/safety are
read automatically from the run's config.json, same as the eval scripts).

  python watch.py runs/none_w0_s0
  python watch.py runs/xcom_w1_s0 --target 0.03
  python watch.py runs/capture_w1_s0 --episodes 5
  python watch.py runs/preliminary_s0
  python watch.py runs/none_w0_s0 --prob 0.02 --force 30   # with perturbation
"""
import argparse
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from postural_env import PosturalEnv, load_run_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--target", type=float, default=None,
                    help="Fixed target_y (ignored for preliminary runs, which "
                         "have no target). Omit for random targets each episode.")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--prob", type=float, default=0.0,
                    help="Per-step probability of a perturbation force. "
                         "0 (default) = clean, no disturbance.")
    p.add_argument("--force", type=float, default=20.0,
                    help="Max perturbation force magnitude (N), used only if --prob > 0.")
    a = p.parse_args()

    cfg = load_run_config(a.run_dir)
    model = PPO.load(f"{a.run_dir}/model")

    target = None if cfg["mode"] == "preliminary" else a.target
    force_range = (-a.force, a.force) if a.prob > 0 else (0, 0)

    print(f"Mode: {cfg['mode']} (safety={cfg['safety']})"
          + (f", perturbation: prob={a.prob}/step, force=+-{a.force}N" if a.prob > 0 else ", clean (no perturbation)"))

    venv = DummyVecEnv([lambda: TimeLimit(
        PosturalEnv(mode=cfg["mode"], safety=cfg["safety"], safety_weight=0.0,
                   fixed_target=target, disturb_prob=a.prob, force_range=force_range,
                   render_mode="human"),
        max_episode_steps=1000)])
    venv = VecNormalize.load(f"{a.run_dir}/vecnormalize.pkl", venv)
    venv.training = False
    venv.norm_reward = False

    for ep in range(a.episodes):
        obs = venv.reset()
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = venv.step(act)
            done = done[0]
        i = info[0]
        if cfg["mode"] == "preliminary":
            print(f"episode {ep+1}: final_com_y={i['com_y']:.4f} failed={i['failed']}")
        else:
            print(f"episode {ep+1}: target={i['target_y']:.4f} "
                  f"final_com_y={i['com_y']:.4f} failed={i['failed']}")

    venv.close()


if __name__ == "__main__":
    main()
