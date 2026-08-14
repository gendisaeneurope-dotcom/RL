"""
cross_subject_baseline_3way.py
================================
Extends the existing 2-subject (003 vs 004) human-vs-human baseline to
all 3 available participants (003, 004, 005). Produces:
  - within-subject correlation for each of the 3 subjects
  - across-subject correlation for each of the 3 pairs (003-004, 003-005, 004-005)
  - a combined mean, to use as the ceiling when ranking candidates


Usage:
    python cross_subject_baseline_3way.py
"""
import random
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

HUMAN_DIR = "C:\\Gepi10\\SPACEMED\\Sem4_Thesis\\Thesis\\Workspace\\Gymnasium_RL\\"
SUBJECT_FILES = {
    "subject003": "human_com_cleaned_subject003_v9.csv",
    "subject004": "human_com_cleaned_subject004_v9.csv",
    "subject005": "human_com_cleaned_subject005_v9.csv",
}
COMX_COL = "com_ml_human"   # task axis
TRIAL_COL = "trial_id"
RESAMPLE_LEN = 60
N_PAIRS = 500


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


def load_trials(path):
    df = pd.read_csv(path, low_memory=False)
    trials = []
    for trial_id, g in df.groupby(TRIAL_COL):
        vals = g[COMX_COL].to_numpy()
        if len(vals) < 5:
            continue
        vals_zeroed = zero_reference(vals)
        net_direction = np.sign(vals_zeroed[-1] - vals_zeroed[0])
        if net_direction > 0:
            vals_zeroed = -vals_zeroed
        trials.append({"trial_id": trial_id, "traj": resample_to_fixed_length(vals_zeroed)})
    return trials


def sample_pairs(trials_a, trials_b, n_pairs, same_subject):
    random.seed(0)
    dtw_vals, corr_vals, rmse_vals = [], [], []
    n_sampled = 0
    attempts = 0
    while n_sampled < n_pairs and attempts < n_pairs * 20:
        attempts += 1
        t1 = random.choice(trials_a)
        t2 = random.choice(trials_b)
        if same_subject and t1["trial_id"] == t2["trial_id"]:
            continue
        dtw_vals.append(dtw_distance(t1["traj"], t2["traj"]))
        corr_vals.append(float(np.corrcoef(t1["traj"], t2["traj"])[0, 1]))
        rmse_vals.append(rmse(t1["traj"], t2["traj"]))
        n_sampled += 1
    return dtw_vals, corr_vals, rmse_vals


def report(label, dtw_vals, corr_vals, rmse_vals):
    print(f"=== {label}, n={len(dtw_vals)} ===")
    print(f"Mean DTW:  {np.mean(dtw_vals):.4f}  (std={np.std(dtw_vals):.4f})")
    print(f"Mean corr: {np.mean(corr_vals):.4f}  (std={np.std(corr_vals):.4f})")
    print(f"Mean RMSE: {np.mean(rmse_vals):.4f} m  (std={np.std(rmse_vals):.4f})")
    print()
    return np.mean(corr_vals), np.mean(dtw_vals)


if __name__ == "__main__":
    trials = {}
    for name, fname in SUBJECT_FILES.items():
        trials[name] = load_trials(HUMAN_DIR + fname)
        print(f"Loaded {len(trials[name])} trials for {name}.")
    print()

    all_corrs, all_dtws = [], []

    print("--- WITHIN-SUBJECT ---")
    for name in SUBJECT_FILES:
        c, d = report(f"WITHIN {name}", *sample_pairs(trials[name], trials[name], N_PAIRS, same_subject=True))
        all_corrs.append(c); all_dtws.append(d)

    print("--- ACROSS-SUBJECT ---")
    names = list(SUBJECT_FILES.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            c, d = report(f"ACROSS {a} vs {b}", *sample_pairs(trials[a], trials[b], N_PAIRS, same_subject=False))
            all_corrs.append(c); all_dtws.append(d)

    print("=" * 60)
    print(f"COMBINED CEILING (mean across all {len(all_corrs)} within+across comparisons):")
    print(f"  Mean correlation: {np.mean(all_corrs):.4f}")
    print(f"  Mean DTW distance: {np.mean(all_dtws):.4f}")
    print("=" * 60)
    print("as the ceiling when ranking candidates for the pose illustration.")
