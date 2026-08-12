"""
compare_resync_comy.py
==========================
  1. Plot com_y (the safety-relevant axis) instead of / alongside com_x.
  2. Evaluate the policy on a SINGLE FIXED TARGET (matching the human
     data's actual target), with small noise on starting position only
     -- not alternating between two targets.
  3. "Stretch" / resynchronize signals: rescale time so each trajectory's
     50%-of-final-displacement point lines up at the same x-axis position
     across human and sim, as a simple visual alternative to DTW.

Usage:
    python compare_resync_comy.py
"""
from unittest import result

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import TimeLimit

N_TIME_BINS = 100
HUMAN_CSV_PATH = "C:\\Gepi10\\SPACEMED\\Sem4_Thesis\\Thesis\\Workspace\\Gymnasium_RL\\human_com_cleaned_subject003_v8.csv"
HUMAN_COMX_COL = "com_x_human"
HUMAN_COMY_COL = "com_y_human" 
HUMAN_TRIAL_COL = "trial_id"
N_SIM_EPISODES = 40

# Set this to the human data's ACTUAL target displacement, not an
# arbitrary +/-0.08. Check your human CSV / experiment protocol for the
# real value before running.
FIXED_TARGET = 0.08
START_POSITION_NOISE_STD = 0.005  # small noise on start, per feedback point 3

CANDIDATES = {
    "candidate1": dict(module="candidate1_ap_comy", 
                       env_class="Candidate1Env",
                       model="ppo_candidate1_ap_comy1_staypenalty6", 
                       vecnorm="vecnormalize_candidate1_ap_comy1_staypenalty6.pkl"),

    "candidate2": dict(module="candidate2_ap_comy1_staypenalty", 
                       env_class="Candidate2Env",
                        model="ppo_candidate2_ap_comy1_staypenalty_6", 
                        vecnorm="vecnormalize_candidate2_ap_comy1_staypenalty_6.pkl"),

    "candidate3": dict(module="candidate3_ap_comy1_staypenalty", 
                       env_class="Candidate3Env",
                       model="ppo_candidate3_ap_comy1_staypenalty6", 
                       vecnorm="vecnormalize_candidate3_ap_comy1_staypenalty6.pkl")
    
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


def find_halfway_index(traj):
    """Index where |traj| first reaches 50% of its own final magnitude."""
    final_mag = abs(traj[-1])
    if final_mag < 1e-9:
        return len(traj) // 2
    target = 0.5 * final_mag
    idx = np.argmax(np.abs(traj) >= target)
    return idx if idx > 0 else len(traj) // 2


def resync_stretch(traj, n_bins=N_TIME_BINS):
    """Rescales time in TWO piecewise-linear segments (before/after the
    halfway point) so the halfway point always lands at normalized time
    0.5, regardless of when it originally occurred. This is the
    'stretch' resynchronization requested -- a simple alternative to DTW."""
    halfway_idx = find_halfway_index(traj)
    if halfway_idx <= 0 or halfway_idx >= len(traj) - 1:
        return resample_to_common_axis(traj, n_bins)

    first_half = traj[:halfway_idx + 1]
    second_half = traj[halfway_idx:]

    t_new = np.linspace(0.0, 1.0, n_bins)
    n_first = int(n_bins * 0.5)
    n_second = n_bins - n_first

    t_first_orig = np.linspace(0.0, 1.0, len(first_half))
    t_first_new = np.linspace(0.0, 1.0, n_first)
    resync_first = np.interp(t_first_new, t_first_orig, first_half)

    t_second_orig = np.linspace(0.0, 1.0, len(second_half))
    t_second_new = np.linspace(0.0, 1.0, n_second)
    resync_second = np.interp(t_second_new, t_second_orig, second_half)

    result = np.concatenate([resync_first, resync_second[1:]])
    if len(result) != n_bins:
        result = resample_to_common_axis(result, n_bins)
    return result


def load_human_data():
    df = pd.read_csv(HUMAN_CSV_PATH, low_memory=False)
    has_comy = HUMAN_COMY_COL in df.columns
    if not has_comy:
        print(f"WARNING: '{HUMAN_COMY_COL}' not found in human CSV. "
              f"com_y overlay will be skipped for human data -- check your "
              f"raw data pipeline for the correct column name (likely "
              f"derived from 'com.1' the same way com_x came from 'com.0').")

    comx_all, comy_all, comx_resync_all = [], [], []
    for trial_id, g in df.groupby(HUMAN_TRIAL_COL):
        vals_x = g[HUMAN_COMX_COL].to_numpy()
        if len(vals_x) < 5:
            continue
        traj_x = zero_reference_and_canonicalize(vals_x)
        comx_all.append(resample_to_common_axis(traj_x))
        comx_resync_all.append(resync_stretch(traj_x))
        if has_comy:
            vals_y = g[HUMAN_COMY_COL].to_numpy()
            comy_all.append(resample_to_common_axis(vals_y - vals_y[0]))

    return (np.array(comx_all), np.array(comy_all) if has_comy else None,
            np.array(comx_resync_all))


def collect_sim_data(module_name, env_class_name, model_path, vecnorm_path,
                      n_episodes=N_SIM_EPISODES):
    module = __import__(module_name)
    EnvClass = getattr(module, env_class_name)
    model = PPO.load(model_path)

    comx_all, comy_all, comx_resync_all = [], [], []
    for ep in range(n_episodes):
        # FIXED TARGET per feedback point 3 -- no alternation.
        env_instance = EnvClass(fixed_target=FIXED_TARGET)
        venv = DummyVecEnv([lambda: TimeLimit(env_instance, max_episode_steps=1000)])
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
        venv.seed(ep)
        obs = venv.reset()

        # Small noise on starting position -- applied via seed variation
        # above; if your env's reset() doesn't already randomize start
        # position slightly, add noise explicitly here matching
        # START_POSITION_NOISE_STD.

        com_x_traj, com_y_traj = [], []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done_v, info = venv.step(action)
            done = bool(done_v[0])
            com_x_traj.append(info[0]["com_x"])
            com_y_traj.append(info[0]["com_y"])
        venv.close()

        traj_x = zero_reference_and_canonicalize(np.array(com_x_traj))
        comx_all.append(resample_to_common_axis(traj_x))
        comx_resync_all.append(resync_stretch(traj_x))
        traj_y = np.array(com_y_traj) - com_y_traj[0]
        comy_all.append(resample_to_common_axis(traj_y))

    return np.array(comx_all), np.array(comy_all), np.array(comx_resync_all)


def plot_comy_overlay(human_comy, sim_comy, name):
    t = np.linspace(0.0, 1.0, N_TIME_BINS)
    fig, ax = plt.subplots(figsize=(10, 6))
    if human_comy is not None:
        for traj in human_comy:
            ax.plot(t, traj, color="gray", alpha=0.15, linewidth=0.8)
        ax.plot(t, human_comy.mean(axis=0), color="black", linewidth=2.5,
                 label=f"Human mean com_y (n={len(human_comy)})")
    for traj in sim_comy:
        ax.plot(t, traj, color="tab:orange", alpha=0.15, linewidth=0.8)
    ax.plot(t, sim_comy.mean(axis=0), color="tab:red", linewidth=2.5,
             label=f"{name} mean com_y (n={len(sim_comy)})")
    ax.set_xlabel("Normalized trial time")
    ax.set_ylabel("Zero-referenced com_y (m) -- safety-relevant axis")
    ax.set_title(f"com_y comparison: human vs {name}")
    ax.legend()
    ax.grid(alpha=0.3)
    out_path = f"comy_overlay_{name.replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_comx_vs_comy(human_comx, human_comy, sim_comx, sim_comy, name):
    fig, ax = plt.subplots(figsize=(8, 8))
    if human_comy is not None:
        for i in range(len(human_comx)):
            ax.plot(human_comx[i], human_comy[i], color="gray", alpha=0.1, linewidth=0.6)
    for i in range(len(sim_comx)):
        ax.plot(sim_comx[i], sim_comy[i], color="tab:orange", alpha=0.15, linewidth=0.6)
    if human_comy is not None:
        ax.plot(human_comx.mean(axis=0), human_comy.mean(axis=0), color="black",
                 linewidth=2.5, label="Human mean")
    ax.plot(sim_comx.mean(axis=0), sim_comy.mean(axis=0), color="tab:red",
             linewidth=2.5, label=f"{name} mean")
    ax.set_xlabel("com_x (m)")
    ax.set_ylabel("com_y (m)")
    ax.set_title(f"com_x vs com_y trajectory: human vs {name}")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    out_path = f"comx_vs_comy_{name.replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_resync_comx(human_resync, sim_resync, name):
    t = np.linspace(0.0, 1.0, N_TIME_BINS)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, human_resync.mean(axis=0), color="black", linewidth=2.5, label="Human mean (resynced)")
    ax.plot(t, sim_resync.mean(axis=0), color="tab:red", linewidth=2.5, label=f"{name} mean (resynced)")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Resync anchor (50% displacement)")
    ax.set_xlabel("Resynchronized normalized time")
    ax.set_ylabel("Zero-referenced com_x (m)")
    ax.set_title(f"Resynchronized comparison: human vs {name}")
    ax.legend()
    ax.grid(alpha=0.3)
    corr, pval = pearsonr(human_resync.mean(axis=0), sim_resync.mean(axis=0))
    ax.text(0.02, 0.02, f"corr={corr:.3f}, p={pval:.2e}", transform=ax.transAxes,
            fontsize=10, va="bottom")
    out_path = f"resync_comx_{name.replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path} (resynced corr={corr:.4f})")
    return corr, pval


if __name__ == "__main__":
    print("Loading human data...")
    human_comx, human_comy, human_resync = load_human_data()
    print(f"Done. n_human_trials={len(human_comx)}\n")

    rows = []
    for name, cfg in CANDIDATES.items():
        print(f"=== {name} ===")
        try:
            sim_comx, sim_comy, sim_resync = collect_sim_data(
                cfg["module"], cfg["env_class"], cfg["model"], cfg["vecnorm"])

            plot_comy_overlay(human_comy, sim_comy, name)
            plot_comx_vs_comy(human_comx, human_comy, sim_comx, sim_comy, name)
            resync_corr, resync_pval = plot_resync_comx(human_resync, sim_resync, name)

            comy_corr, comy_pval = (pearsonr(human_comy.mean(axis=0), sim_comy.mean(axis=0))
                                     if human_comy is not None else (None, None))

            print(f"  Resynced com_x correlation: {resync_corr:.4f} (p={resync_pval:.2e})")
            if comy_corr is not None:
                print(f"  com_y correlation: {comy_corr:.4f} (p={comy_pval:.2e})\n")

            rows.append({"candidate": name, "resync_comx_corr": round(resync_corr, 4),
                         "comy_corr": round(comy_corr, 4) if comy_corr is not None else None,
                         "status": "OK"})
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}\n")
            rows.append({"candidate": name, "resync_comx_corr": None,
                         "comy_corr": None, "status": f"FAILED: {e}"})

    summary = pd.DataFrame(rows)
    summary.to_csv("resync_comy_summary.csv", index=False)
    print("=== SUMMARY ===")
    print(summary.to_string(index=False))
    print("\nSaved to resync_comy_summary.csv")
