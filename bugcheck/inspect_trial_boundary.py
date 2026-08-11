"""Inspect what current_state actually looks like inside trial_group 227
(right) and trial_group 8 (left) -- the two trials the trajectory plot
flagged as suspicious (spending most of their duration near the OPPOSITE
side's value before swinging to target only at the very end).

Run on the RAW csv (the one with a current_state column), not the already-
extracted files.

  python inspect_trial_boundary.py "C:\Gepi10\SPACEMED\Sem4_Thesis\Thesis\Workspace\Gymnasium_RL\resynchronized_data_subject003.csv"
"""
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path, low_memory=False)

state = df["current_state"].astype(str)
df["trial_id"] = (state != state.shift()).cumsum()

task_mask = state.isin(["GO_TO_LEFT_CIRCLE_AFTER_TRIAL", "GO_TO_RIGHT_CIRCLE_AFTER_TRIAL",
                        "STAY_IN_LEFT_CIRCLE", "STAY_IN_RIGHT_CIRCLE"])
task_df = df[task_mask].copy()
task_df["side"] = task_df["current_state"].str.contains("LEFT").map({True: "left", False: "right"})
task_df["trial_group"] = (task_df["side"] != task_df["side"].shift()).cumsum()

for tg in [308]:
    g = task_df[task_df.trial_group == tg].sort_values("t_rel")
    print(f"=== trial_group {tg} ===")
    print(f"  n rows: {len(g)}, t_rel range: {g['t_rel'].min():.2f} to {g['t_rel'].max():.2f}")
    print(f"  current_state value counts (in order of first appearance):")
    seen = []
    for s in g["current_state"]:
        if not seen or seen[-1] != s:
            seen.append(s)
    for s in seen:
        n = (g["current_state"] == s).sum()
        t_first = g[g["current_state"] == s]["t_rel"].min()
        t_last = g[g["current_state"] == s]["t_rel"].max()
        print(f"    {s:35s} n={n:4d}  t_rel {t_first:.2f}-{t_last:.2f}")
    print(f"  com.0 at start: {g['com.0'].iloc[0]:.1f}, at end: {g['com.0'].iloc[-1]:.1f}")
    print()
