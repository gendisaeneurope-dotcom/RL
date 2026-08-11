"""
Runs the sim-vs-human comparison across all candidate configs, using the
corrected v6 human baseline, to see if any candidate comes closer to the
human-vs-human baseline (corr ~0.83, DTW ~0.005) than the others.

Usage:
    python compare_all_candidates.py
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

HUMAN_CSV_PATH = r"C:\Gepi10\SPACEMED\Sem4_Thesis\Thesis\Workspace\Gymnasium_RL\human_com_cleaned_subject003_v7.csv"
HUMAN_COMX_COL = "com_x_human"
HUMAN_TRIAL_COL = "trial_id"
RESAMPLE_LEN = 60
N_SIM_EPISODES = 40

CONFIGS = CONFIGS = {
    "candidate1": dict(module="candidate1_ap_comy", cls="Candidate1Env",
                        model="ppo_candidate1_ap_comy1_staypenalty6",
                        vecnorm="vecnormalize_candidate1_ap_comy1_staypenalty6.pkl",
                         color="#1F77B4"),
    "candidate2": dict(module="candidate2_ap_comy1_staypenalty_jointfix", cls="Candidate2Env",
                        model="ppo_candidate2_ap_comy1_staypenalty_jointfix",
                        vecnorm="vecnormalize_candidate2_ap_comy1_staypenalty_jointfix.pkl",
                         color="#2CA02C"),
    "candidate3": dict(module="candidate3_ap_comy1_staypenalty", cls="Candidate3Env",
                        model="ppo_candidate3_ap_comy1_staypenalty6",
                        vecnorm="vecnormalize_candidate3_ap_comy1_staypenalty6.pkl",
                         color="#9467BD"),
}


def resample_to_fixed_length(traj, length=RESAMPLE_LEN):
    traj = np.asarray(traj, dtype=float)
    old_x = np.linspace(0, 1, len(traj))
    new_x = np.linspace(0, 1, length)
    return np.interp(new_x, old_x, traj)


def zero_reference(traj):
    traj = np.asarray(traj, dtype=float)
    return traj - traj[0]


def dtw_distance(a, b):
    a = np.asarray(a, dtype=float).reshape(-1, 1)
    b = np.asarray(b, dtype=float).reshape(-1, 1)
    dist, _ = fastdtw(a, b, dist=euclidean)
    return dist / len(a)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load_human_trials():
    df = pd.read_csv(HUMAN_CSV_PATH, low_memory=False)
    trials = []
    for trial_id, g in df.groupby(HUMAN_TRIAL_COL):
        vals = g[HUMAN_COMX_COL].to_numpy()
        if len(vals) < 5:
            continue
        vals_zeroed = zero_reference(vals)
        net_direction = np.sign(vals_zeroed[-1] - vals_zeroed[0])
        if net_direction > 0:
            vals_zeroed = -vals_zeroed
        trials.append({"trial_id": trial_id, "traj": resample_to_fixed_length(vals_zeroed)})
    return trials


def collect_sim_trajectories(cfg_key, n_episodes=N_SIM_EPISODES):
    cfg = CONFIGS[cfg_key]
    module = __import__(cfg["module"])
    EnvClass = getattr(module, cfg["cls"])
    model = PPO.load(cfg["model"])

    sims = []
    for ep in range(n_episodes):
        target = 0.08 if ep % 2 == 0 else -0.08
        venv = DummyVecEnv([lambda: TimeLimit(EnvClass(fixed_target=target), max_episode_steps=1000)])
        venv = VecNormalize.load(cfg["vecnorm"], venv)
        venv.training = False
        venv.norm_reward = False
        venv.seed(ep)
        obs = venv.reset()

        com_x_traj, target_x = [], None
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done_v, info = venv.step(action)
            done = bool(done_v[0])
            com_x_traj.append(float(info[0]["com_x"]))
            if target_x is None:
                target_x = float(info[0]["target_x"])
        venv.close()

        net_direction = np.sign(target_x)
        com_x_traj_zeroed = zero_reference(com_x_traj)
        if net_direction > 0:
            com_x_traj_zeroed = -com_x_traj_zeroed
        sims.append({"episode": ep, "traj": resample_to_fixed_length(com_x_traj_zeroed)})
    return sims


if __name__ == "__main__":
    trials = load_human_trials()
    print(f"Loaded {len(trials)} human trials.\n")
    print("Human-vs-human baseline for reference: corr ~0.83, DTW ~0.005\n")

    for cfg_key in CONFIGS:
        try:
            sims = collect_sim_trajectories(cfg_key)
        except Exception as e:
            print(f"{cfg_key}: SKIPPED ({type(e).__name__}: {e})\n")
            continue

        dtw_vals, corr_vals, rmse_vals = [], [], []
        for t in trials:
            dtws = [dtw_distance(t["traj"], s["traj"]) for s in sims]
            corrs = [np.corrcoef(t["traj"], s["traj"])[0, 1] for s in sims]
            rmses = [rmse(t["traj"], s["traj"]) for s in sims]
            best_idx = int(np.argmin(dtws))
            dtw_vals.append(dtws[best_idx])
            corr_vals.append(corrs[best_idx])
            rmse_vals.append(rmses[best_idx])

        print(f"=== {cfg_key} ===")
        print(f"  mean best-match corr: {np.mean(corr_vals):.4f}")
        print(f"  mean best-match DTW:  {np.mean(dtw_vals):.4f}")
        print(f"  mean best-match RMSE: {np.mean(rmse_vals):.4f} m")
        print(f"  (human baseline: corr 0.83, DTW 0.005)")
        print()