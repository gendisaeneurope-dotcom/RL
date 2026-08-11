"""
compare_mean_trajectories.py
================================

WHAT THIS CHANGES VS compare_sim_vs_human_rmse_v2.py / compare_all_candidates.py
---------------------------------------------------------------------------------
Those scripts compute RMSE/correlation/DTW for every (human trial, sim
episode) PAIR individually -- hundreds of noisy single-trial-vs-single-
trial comparisons. This script instead:

  1. Resamples every trial (human AND sim) onto a common normalized time
     axis (0 to 1), same direction-canonicalization and zero-referencing
     already established as correct in the existing pipeline.
  2. Computes the MEAN trajectory (and SEM band) across ALL human trials,
     and separately the MEAN trajectory across ALL sim episodes.
  3. Computes ONE correlation and ONE DTW distance between the two MEAN
     curves. Averaging trajectories BEFORE correlating removes most
     of the trial-to-trial noise; averaging correlation VALUES afterward
     (what the old scripts did) does not.
  4. Produces the overlay ("spaghetti") plot explicitly requested: every
     individual human trial in light gray, every individual sim episode
     in light color, with the two bold mean curves drawn on top and a
     shaded +/-1 SEM band around each mean. This is the plot needed to
     visually confirm whether noise or genuine mismatch is driving any
     remaining low correlation.

     
Usage:
    python compare_mean_trajectories.py candidate2_ap_comy1_staypenalty_jointfix
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import TimeLimit

N_TIME_BINS = 100  # normalized time axis resolution, 0..1
HUMAN_CSV_PATH = "C:\\Gepi10\\SPACEMED\\Sem4_Thesis\\Thesis\\Workspace\\Gymnasium_RL\\human_com_cleaned_subject003_v7.csv"
HUMAN_COMX_COL = "com_x_human"
HUMAN_TRIAL_COL = "trial_id"
N_SIM_EPISODES = 40


module_name = "candidate1_ap_comy_jointfix"  # for candidate 1
model_path = "ppo_candidate1_ap_comy1_staypenalty_jointfix"
vecnorm_path = "vecnormalize_candidate1_ap_comy1_staypenalty_jointfix.pkl"

# for candidate 3
module_name = "candidate3_ap_comy1_staypenalty6_jointfix"
model_path = "ppo_candidate3_ap_comy1_staypenalty_jointfix"
vecnorm_path = "vecnormalize_candidate3_ap_comy1_staypenalty_jointfix.pkl"

def resample_to_common_axis(traj, n_bins=N_TIME_BINS):
    """Resample a 1D trajectory of arbitrary length onto n_bins evenly
    spaced points over normalized time [0, 1]."""
    t_orig = np.linspace(0.0, 1.0, len(traj))
    t_new = np.linspace(0.0, 1.0, n_bins)
    return np.interp(t_new, t_orig, traj)


def zero_reference_and_canonicalize(traj):
    """Same convention already established in the pipeline: subtract the
    trial's own starting value, then flip sign if net displacement is
    positive, so every trial ends up moving in the same canonical
    direction regardless of which way the original target was."""
    traj = traj - traj[0]
    if traj[-1] > 0:
        traj = -traj
    return traj


def load_human_mean_and_all(csv_path=HUMAN_CSV_PATH, comx_col=HUMAN_COMX_COL,
                             trial_col=HUMAN_TRIAL_COL):
    df = pd.read_csv(csv_path, low_memory=False)
    all_resampled = []
    for trial_id, g in df.groupby(trial_col):
        vals = g[comx_col].to_numpy()
        if len(vals) < 5:
            continue
        traj = zero_reference_and_canonicalize(vals)
        all_resampled.append(resample_to_common_axis(traj))
    all_resampled = np.array(all_resampled)  # (n_trials, N_TIME_BINS)
    mean_traj = all_resampled.mean(axis=0)
    sem_traj = all_resampled.std(axis=0) / np.sqrt(len(all_resampled))
    return mean_traj, sem_traj, all_resampled


def collect_sim_mean_and_all(module_name, env_class_name, model_path, vecnorm_path,
                              n_episodes=N_SIM_EPISODES):
    module = __import__(module_name)
    EnvClass = getattr(module, env_class_name)
    model = PPO.load(model_path)

    all_resampled = []
    for ep in range(n_episodes):
        target = 0.08 if ep % 2 == 0 else -0.08
        env_instance = EnvClass(fixed_target=target)
        venv = DummyVecEnv([lambda: TimeLimit(env_instance, max_episode_steps=1000)])
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
        venv.seed(ep)
        obs = venv.reset()
        com_x_traj = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done_v, info = venv.step(action)
            done = bool(done_v[0])
            com_x_traj.append(info[0]["com_x"])
        venv.close()
        traj = zero_reference_and_canonicalize(np.array(com_x_traj))
        all_resampled.append(resample_to_common_axis(traj))

    all_resampled = np.array(all_resampled)
    mean_traj = all_resampled.mean(axis=0)
    sem_traj = all_resampled.std(axis=0) / np.sqrt(len(all_resampled))
    return mean_traj, sem_traj, all_resampled


def compare_means(human_mean, sim_mean):
    """The single, robust correlation/RMSE this whole script exists to
    produce -- computed ONCE, on the two averaged curves, not averaged
    afterward from many noisy pairs."""
    corr, pval = pearsonr(human_mean, sim_mean)
    rmse = float(np.sqrt(np.mean((human_mean - sim_mean) ** 2)))
    return corr, pval, rmse


def plot_overlay(human_all, human_mean, human_sem, sim_all, sim_mean, sim_sem,
                  candidate_name, out_path=None):
    t = np.linspace(0.0, 1.0, N_TIME_BINS)
    fig, ax = plt.subplots(figsize=(10, 6))

    for traj in human_all:
        ax.plot(t, traj, color="gray", alpha=0.15, linewidth=0.8)
    for traj in sim_all:
        ax.plot(t, traj, color="tab:orange", alpha=0.15, linewidth=0.8)

    ax.plot(t, human_mean, color="black", linewidth=2.5, label=f"Human mean (n={len(human_all)})")
    ax.fill_between(t, human_mean - human_sem, human_mean + human_sem, color="black", alpha=0.2)

    ax.plot(t, sim_mean, color="tab:red", linewidth=2.5, label=f"{candidate_name} mean (n={len(sim_all)})")
    ax.fill_between(t, sim_mean - sim_sem, sim_mean + sim_sem, color="tab:red", alpha=0.2)

    ax.set_xlabel("Normalized trial time (0 = start, 1 = end)")
    ax.set_ylabel("Zero-referenced, direction-canonicalized com_x (m)")
    ax.set_title(f"All trials (light) + mean ± SEM (bold): human vs {candidate_name}")
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = out_path or f"overlay_mean_comparison_{candidate_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved overlay plot to {out_path}")


if __name__ == "__main__":
    candidate_tag = "Candidate 3" 

    print("Loading and averaging human trials...")
    human_mean, human_sem, human_all = load_human_mean_and_all()

    module_name = "candidate3_ap_comy1_staypenalty6_jointfix"
    env_class_name = "Candidate3Env"
    model_path = "ppo_candidate3_ap_comy1_staypenalty_jointfix"
    vecnorm_path = "vecnormalize_candidate3_ap_comy1_staypenalty_jointfix.pkl"

    print(f"Loading and averaging sim episodes for {candidate_tag}...")
    sim_mean, sim_sem, sim_all = collect_sim_mean_and_all(
        module_name, env_class_name, model_path, vecnorm_path)

    corr, pval, rmse = compare_means(human_mean, sim_mean)
    print(f"\nCorrelation: {corr:.4f} (p={pval:.2e}), RMSE: {rmse:.5f} m")
    plot_overlay(human_all, human_mean, human_sem, sim_all, sim_mean, sim_sem, candidate_tag)
    print(f"RMSE between mean curves: {rmse:.5f} m")


    plot_overlay(human_all, human_mean, human_sem, sim_all, sim_mean, sim_sem, candidate_tag)
