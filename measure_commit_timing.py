"""
Finds, for each human trial, the time at which the subject first commits
to the movement (crosses 10% of total displacement) and how that compares
to total trial duration. Used to pick a defensible TRACKING_DELAY_STEPS
value instead of guessing.

Usage:
    python measure_commit_timing.py
"""
import pandas as pd
import numpy as np

PATH = "C:\\Gepi10\\SPACEMED\\Sem4_Thesis\\Thesis\\Workspace\\Gymnasium_RL\\human_com_cleaned_subject003_v7.csv"
COMX_COL = "com_x_human"
TRIAL_COL = "trial_id"
THRESHOLD_FRAC = 0.05  # % of total displacement counted as "committed"

df = pd.read_csv(PATH, low_memory=False)

commit_times, commit_fracs, durations = [], [], []

for trial_id, g in df.groupby(TRIAL_COL):
    g = g.sort_values("t_rel")
    vals = g[COMX_COL].to_numpy()
    t = g["t_rel"].to_numpy()
    if len(vals) < 5:
        continue

    vals0 = vals - vals[0]
    t0 = t - t[0]
    total_disp = vals0[-1]
    if abs(total_disp) < 1e-6:
        continue

    threshold = THRESHOLD_FRAC * total_disp
    if total_disp > 0:
        crossed = np.where(vals0 >= threshold)[0]
    else:
        crossed = np.where(vals0 <= threshold)[0]
    if len(crossed) == 0:
        continue

    idx = crossed[0]
    duration = t0[-1]
    if duration <= 0:
        continue

    commit_times.append(t0[idx])
    commit_fracs.append(t0[idx] / duration)
    durations.append(duration)

commit_times = np.array(commit_times)
commit_fracs = np.array(commit_fracs)
durations = np.array(durations)

print(f"n trials analyzed: {len(commit_times)}")
print(f"Mean trial duration: {durations.mean():.3f} s")
print(f"Mean time-to-commit (5% threshold): {commit_times.mean():.3f} s")
print(f"Mean commit fraction of trial duration: {commit_fracs.mean():.3f}")
print(f"Median commit fraction: {np.median(commit_fracs):.3f}")