"""
Within-subject vs across-subject null baselines, using human_com_cleaned_
subject003_v6.csv and human_com_cleaned_subject004_v6.csv.

Usage:
    python cross_subject_baseline.py
"""
import random
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

PATH_003 = r"C:\Gepi10\SPACEMED\Sem4_Thesis\Thesis\Workspace\Gymnasium_RL\human_com_cleaned_subject003_v7.csv"
PATH_004 = r"C:\Gepi10\SPACEMED\Sem4_Thesis\Thesis\Workspace\Gymnasium_RL\human_com_cleaned_subject004_v7.csv"
COMX_COL = "com_x_human"
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
        trials.append({
            "trial_id": trial_id,
            "traj": resample_to_fixed_length(vals_zeroed),
        })
    return trials


def sample_pairs(trials_a, trials_b, n_pairs, same_subject):
    random.seed(0)
    dtw_vals, corr_vals, rmse_vals = [], [], []
    n_sampled = 0
    while n_sampled < n_pairs:
        t1 = random.choice(trials_a)
        t2 = random.choice(trials_b)
        if same_subject and t1["trial_id"] == t2["trial_id"]:
            continue
        dtw_vals.append(dtw_distance(t1["traj"], t2["traj"]))
        corr_vals.append(np.corrcoef(t1["traj"], t2["traj"])[0, 1])
        rmse_vals.append(rmse(t1["traj"], t2["traj"]))
        n_sampled += 1
    return dtw_vals, corr_vals, rmse_vals


def report(label, dtw_vals, corr_vals, rmse_vals):
    print(f"=== {label}, n={len(dtw_vals)} ===")
    print(f"Mean DTW:  {np.mean(dtw_vals):.4f}  (std={np.std(dtw_vals):.4f})")
    print(f"Mean corr: {np.mean(corr_vals):.4f}  (std={np.std(corr_vals):.4f})")
    print(f"Mean RMSE: {np.mean(rmse_vals):.4f} m  (std={np.std(rmse_vals):.4f})")
    print()


if __name__ == "__main__":
    trials_003 = load_trials(PATH_003)
    trials_004 = load_trials(PATH_004)
    print(f"Loaded {len(trials_003)} trials for subject003, {len(trials_004)} for subject004.\n")

    report("WITHIN subject003", *sample_pairs(trials_003, trials_003, N_PAIRS, same_subject=True))
    report("WITHIN subject004", *sample_pairs(trials_004, trials_004, N_PAIRS, same_subject=True))
    report("ACROSS subject003 vs subject004", *sample_pairs(trials_003, trials_004, N_PAIRS, same_subject=False))

    print("Compare all three against your sim-vs-human numbers (corr ~0.11-0.18, DTW ~0.0065-0.0415).")
    print("If ACROSS is much worse (lower corr, higher DTW) than WITHIN, that's real")
    print("evidence people move differently from each other -- and tells you whether")
    print("the sim is closer to 'matching a specific person' or 'matching human movement in general.'")