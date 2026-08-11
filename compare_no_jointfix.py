"""
compare_no_jointfix.py
==========================
The PURE, ARDITI-FAITHFUL comparison: no joint-limit-aware penalty on any
candidate. This is the correct baseline set for the main results table --
the jointfix versions are a separate, later experiment (your own fix,
motivated by Candidate 1's diagnosed failure), not part of replicating
Arditi et al.'s original formulation.

CONFIRMED from your own pasted source code / dir listing:
  Candidate 1: module=candidate1_ap_comy, class=Candidate1Env,
               model=ppo_candidate1_ap_comy1_staypenalty6,
               vecnorm=vecnormalize_candidate1_ap_comy1_staypenalty6.pkl
  Candidate 3: module=candidate3_ap_comy1_staypenalty, class=Candidate3Env,
               model=ppo_candidate3_ap_comy1_staypenalty6,
               vecnorm=vecnormalize_candidate3_ap_comy1_staypenalty6.pkl

NOT YET CONFIRMED (best guess, following the same naming pattern as
Candidates 1 and 3 -- if wrong, this candidate alone will print a clear
error and the script keeps going for the other two):
  Candidate 2: module=candidate2_ap_comy1_staypenalty, class=Candidate2Env,
               model=ppo_candidate2_ap_comy1_staypenalty6,
               vecnorm=vecnormalize_candidate2_ap_comy1_staypenalty6.pkl

Run once:
    python compare_no_jointfix.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import TimeLimit

N_TIME_BINS = 100
HUMAN_CSV_PATH = r"C:\Gepi10\SPACEMED\Sem4_Thesis\Thesis\Workspace\Gymnasium_RL\human_com_cleaned_subject003_v7.csv"
HUMAN_COMX_COL = "com_x_human"
HUMAN_TRIAL_COL = "trial_id"
N_SIM_EPISODES = 40

CANDIDATES = {
    "Candidate 1 (no jointfix)": dict(
        module="candidate1_ap_comy",
        env_class="Candidate1Env",
        model="ppo_candidate1_ap_comy1_staypenalty6",
        vecnorm="vecnormalize_candidate1_ap_comy1_staypenalty6.pkl",
    ),
    "Candidate 2 (no jointfix)": dict(
    module="candidate2_ap_comy1_staypenalty",
    env_class="Candidate2Env",
    model="ppo_candidate2_ap_comy1_staypenalty_6",
    vecnorm="vecnormalize_candidate2_ap_comy1_staypenalty_6.pkl",
    ),
    "Candidate 3 (no jointfix)": dict(
        module="candidate3_ap_comy1_staypenalty",
        env_class="Candidate3Env",
        model="ppo_candidate3_ap_comy1_staypenalty6",
        vecnorm="vecnormalize_candidate3_ap_comy1_staypenalty6.pkl",
    ),
}


def resample_to_common_axis(traj, n_bins=N_TIME_BINS):
    t_orig = np.linspace(0.0, 1.0, len(traj))
    t_new = np.linspace(0.0, 1.0, n_bins)
    return np.interp(t_new, t_orig, traj)


def zero_reference_and_canonicalize(traj):
    traj = traj - traj[0]
    if traj[-1] > 0:
        traj = -traj
    return traj


def load_human_mean_and_all():
    df = pd.read_csv(HUMAN_CSV_PATH, low_memory=False)
    all_resampled = []
    for trial_id, g in df.groupby(HUMAN_TRIAL_COL):
        vals = g[HUMAN_COMX_COL].to_numpy()
        if len(vals) < 5:
            continue
        traj = zero_reference_and_canonicalize(vals)
        all_resampled.append(resample_to_common_axis(traj))
    all_resampled = np.array(all_resampled)
    mean_traj = all_resampled.mean(axis=0)
    sem_traj = all_resampled.std(axis=0) / np.sqrt(len(all_resampled))
    return mean_traj, sem_traj, all_resampled


def collect_sim_mean_and_all(module_name, env_class_name, model_path, vecnorm_path,
                              n_episodes=N_SIM_EPISODES):
    module = __import__(module_name)
    EnvClass = getattr(module, env_class_name)
    model = PPO.load(model_path)

    all_resampled = []
    n_failed = 0
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
            if info[0].get("failed"):
                n_failed += 1
        venv.close()
        traj = zero_reference_and_canonicalize(np.array(com_x_traj))
        all_resampled.append(resample_to_common_axis(traj))

    all_resampled = np.array(all_resampled)
    mean_traj = all_resampled.mean(axis=0)
    sem_traj = all_resampled.std(axis=0) / np.sqrt(len(all_resampled))
    # Note: n_failed counts steps flagged failed=True at least once, not
    # unique failed episodes -- reported here as a rough flag only. For
    # a precise per-episode failure rate, check episode length < 1000.
    return mean_traj, sem_traj, all_resampled


def plot_overlay(human_all, human_mean, human_sem, sim_all, sim_mean, sim_sem, name):
    t = np.linspace(0.0, 1.0, N_TIME_BINS)
    fig, ax = plt.subplots(figsize=(10, 6))
    for traj in human_all:
        ax.plot(t, traj, color="gray", alpha=0.15, linewidth=0.8)
    for traj in sim_all:
        ax.plot(t, traj, color="tab:orange", alpha=0.15, linewidth=0.8)
    ax.plot(t, human_mean, color="black", linewidth=2.5, label=f"Human mean (n={len(human_all)})")
    ax.fill_between(t, human_mean - human_sem, human_mean + human_sem, color="black", alpha=0.2)
    ax.plot(t, sim_mean, color="tab:red", linewidth=2.5, label=f"{name} mean (n={len(sim_all)})")
    ax.fill_between(t, sim_mean - sim_sem, sim_mean + sim_sem, color="tab:red", alpha=0.2)
    ax.set_xlabel("Normalized trial time (0 = start, 1 = end)")
    ax.set_ylabel("Zero-referenced, direction-canonicalized com_x (m)")
    ax.set_title(f"All trials (light) + mean +/- SEM (bold): human vs {name}")
    ax.legend()
    ax.grid(alpha=0.3)
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    out_path = f"overlay_{safe_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    print("Loading and averaging human trials (shared across all three)...")
    human_mean, human_sem, human_all = load_human_mean_and_all()
    print(f"Done. n_human_trials={len(human_all)}\n")

    rows = []
    for name, cfg in CANDIDATES.items():
        print(f"=== {name} ===")
        try:
            sim_mean, sim_sem, sim_all = collect_sim_mean_and_all(
                cfg["module"], cfg["env_class"], cfg["model"], cfg["vecnorm"])
            corr, pval = pearsonr(human_mean, sim_mean)
            rmse = float(np.sqrt(np.mean((human_mean - sim_mean) ** 2)))
            plot_overlay(human_all, human_mean, human_sem, sim_all, sim_mean, sim_sem, name)
            print(f"  Correlation: {corr:.4f} (p={pval:.2e})")
            print(f"  RMSE: {rmse:.5f} m\n")
            rows.append({"candidate": name, "correlation": round(corr, 4),
                         "p_value": pval, "rmse_m": round(rmse, 5), "status": "OK"})
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            print(f"  Skipping {name}, continuing with remaining candidates.\n")
            rows.append({"candidate": name, "correlation": None,
                         "p_value": None, "rmse_m": None,
                         "status": f"FAILED: {type(e).__name__}: {e}"})

    summary = pd.DataFrame(rows)
    summary.to_csv("no_jointfix_comparison_summary.csv", index=False)
    print("=== FINAL SUMMARY (no-jointfix) ===")
    print(summary.to_string(index=False))
    print("\nSaved to no_jointfix_comparison_summary.csv")
    print("If Candidate 2 shows FAILED, run: dir candidate2*.py")
