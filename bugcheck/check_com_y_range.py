"""
Check the untracked front-back axis (com_y) across episodes for all three
candidates -- confirms or refutes the -0.5 finding from the xcom_vs_boundary
/ joint_angles_vs_com plots.

Usage:
    python check_com_y_range.py
"""
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

CONFIGS = {
    "candidate1_F_ap": dict(module="candidate1_F_ap", cls="Candidate1Env",
                             model="ppo_candidate1_F_ap", vecnorm="vecnormalize_candidate1_F_ap.pkl",
                             log_dir="./training_logs_candidate1_F_ap/"),
    "candidate1_ap_start": dict(module="candidate1_F_ap", cls="Candidate1Env",
                                 model="ppo_candidate1_ap_start", vecnorm="vecnormalize_candidate1_ap_start.pkl",
                                 log_dir="./training_logs_candidate1_ap_start/"),
    "candidate2_ap": dict(module="candidate2_ap", cls="Candidate2Env",
                           model="ppo_candidate2_ap", vecnorm="vecnormalize_candidate2_ap.pkl",
                           log_dir="./training_logs_candidate2_ap/"),
    "candidate2_ap_start": dict(module="candidate2_ap", cls="Candidate2Env",
                                 model="ppo_candidate2_ap_start", vecnorm="vecnormalize_candidate2_ap_start.pkl",
                                 log_dir="./training_logs_candidate2_ap_start/"),
    "candidate2_ap_sw005_01disturb": dict(module="candidate2_ap_disturb", cls="Candidate2Env",
                                            model="ppo_candidate2_ap_sw005_01disturb", vecnorm="vecnormalize_candidate2_ap_sw005_01disturb.pkl",
                                            log_dir="./training_logs_candidate2_ap_sw005_01disturb/"),
    "candidate3_ap": dict(module="candidate3_ap", cls="Candidate3Env",
                           model="ppo_candidate3_ap", vecnorm="vecnormalize_candidate3_ap.pkl",
                           log_dir="./training_logs_candidate3_ap/"),
    "candidate3_ap_start": dict(module="candidate3_ap", cls="Candidate3Env",
                                 model="ppo_candidate3_ap_start", vecnorm="vecnormalize_candidate3_ap_start.pkl",
                                 log_dir="./training_logs_candidate3_ap_start/"),
}


def check(cfg_key, n_episodes=10):
    cfg = CONFIGS[cfg_key]
    module = __import__(cfg["module"])
    EnvClass = getattr(module, cfg["cls"])
    model = PPO.load(cfg["model"])

    all_com_y = []
    for ep in range(n_episodes):
        venv = DummyVecEnv([lambda: TimeLimit(EnvClass(), max_episode_steps=1000)])
        venv = VecNormalize.load(cfg["vecnorm"], venv)
        venv.training = False
        venv.norm_reward = False
        venv.seed(ep)
        obs = venv.reset()

        raw = venv.venv.envs[0].unwrapped
        com_x0, com_y0 = raw._com_xy()

        traj_y = [com_y0]
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done_v, info = venv.step(action)
            done = bool(done_v[0])
            traj_y.append(float(info[0].get("com_y", raw._com_xy()[1])))
        venv.close()
        all_com_y.append(traj_y)

    flat = np.concatenate([np.array(t) for t in all_com_y])
    print(f"\n=== {cfg_key} ===")
    print(f"com_y across {n_episodes} episodes, {len(flat)} total steps:")
    print(f"  mean={flat.mean():.4f}  std={flat.std():.4f}  "
          f"range=[{flat.min():.4f}, {flat.max():.4f}]")
    print(f"  first-step values (reset): {[round(t[0],4) for t in all_com_y]}")
    print(f"  last-step values (final):  {[round(t[-1],4) for t in all_com_y]}")


if __name__ == "__main__":
    for key in CONFIGS:
        try:
            check(key)
        except Exception as e:
            print(f"\nSkipping {key}: {type(e).__name__}: {e}")
