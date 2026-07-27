"""Escalating perturbation robustness check. Ported from the original
eval_perturbation_robustness.py -- same 5 conditions, same two-part report
(reward/length summary, then explicit target-vs-final-com_y per episode).

Model is always loaded as trained WITHOUT perturbation; disturbance is
introduced here, at test time only.

  python eval_perturbation_robustness.py runs/none_w0_s0
"""
import sys
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from postural_env import PosturalEnv, load_run_config

# Same escalating conditions as the original script.
CONDITIONS = [
    {"disturb_prob": 0.0, "force_range": (-20, 20)},     # baseline, no disturbance
    {"disturb_prob": 0.05, "force_range": (-20, 20)},    # light, occasional
    {"disturb_prob": 0.3, "force_range": (-20, 20)},     # moderate, same force
    {"disturb_prob": 1.0, "force_range": (-20, 20)},     # constant, same force
    {"disturb_prob": 1.0, "force_range": (-100, 100)},   # constant, much stronger
]


def make_venv(run_dir, cfg, cond, seed=None):
    def _f():
        e = PosturalEnv(mode=cfg["mode"], safety=cfg["safety"], safety_weight=0.0, **cond)
        e = TimeLimit(e, max_episode_steps=1000)
        if seed is not None:
            e.reset(seed=seed)
        return e
    venv = DummyVecEnv([_f])
    venv = VecNormalize.load(f"{run_dir}/vecnormalize.pkl", venv)
    venv.training = False
    venv.norm_reward = False
    return venv


def main():
    run_dir = sys.argv[1]
    cfg = load_run_config(run_dir)
    model = PPO.load(f"{run_dir}/model")
    is_target_mode = cfg["mode"] != "preliminary"

    for cond in CONDITIONS:
        print(f"\n=== disturb_prob={cond['disturb_prob']}, "
              f"force_range={cond['force_range']} ===")

        # Reward/length summary
        venv = make_venv(run_dir, cfg, cond)
        rewards, lengths = evaluate_policy(model, venv, n_eval_episodes=20,
                                           return_episode_rewards=True)
        venv.close()
        print(f"Mean reward: {np.mean(rewards):.4f} +/- {np.std(rewards):.4f}  |  "
              f"episodes at full length: {sum(1 for l in lengths if l == 1000)}/20")

        if not is_target_mode:
            continue  # preliminary has no target to check against

        # Does it still reach the target despite the disturbance? Reward/
        # length alone can't answer this -- need final com_y vs target_y.
        errors = []
        for ep in range(10):
            venv2 = make_venv(run_dir, cfg, cond, seed=ep)
            obs = venv2.reset()
            info = [{}]
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done_v, info = venv2.step(action)
                done = bool(done_v[0])
            i = info[0]
            err = i["com_y"] - i["target_y"]
            errors.append(err)
            print(f"  target_y={i['target_y']:.4f}, final com_y={i['com_y']:.4f}, "
                  f"error={err:.4f}, failed={i['failed']}")
            venv2.close()
        print(f"Mean |error| across 10 episodes: {np.mean(np.abs(errors)):.4f}")


if __name__ == "__main__":
    main()
