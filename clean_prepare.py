"""


Usage:
    python clean_and_prepare.py subject003
    python clean_and_prepare.py subject004
"""
import sys
import pandas as pd
import numpy as np

subject = sys.argv[1]
RAW_PATH = f"C:\\Gepi10\\SPACEMED\\Sem4_Thesis\\Thesis\\Workspace\\Gymnasium_RL\\resynchronized_data_{subject}.csv"
OUTPUT_PATH = f"C:\\Gepi10\\SPACEMED\\Sem4_Thesis\\Thesis\\Workspace\\Gymnasium_RL\\human_com_cleaned_{subject}_v7.csv"

df = pd.read_csv(RAW_PATH, low_memory=False)

df["com_x_human"] = -df["com.0"] / 1000.0

task_mask = df["current_state"].isin(
    ["GO_TO_LEFT_CIRCLE_AFTER_TRIAL", "GO_TO_RIGHT_CIRCLE_AFTER_TRIAL",
     "STAY_IN_LEFT_CIRCLE", "STAY_IN_RIGHT_CIRCLE"])

# Filter extreme com.0 values (sensor glitches / trailing artifacts) --
# threshold chosen empirically from subject003's washout-block outliers
# (median ~-8, 25-75% range roughly -60 to +55; max was 1555, clearly
# not physiological). Not a principled physiological limit -- an
# eyeballed cutoff. See handoff notes / Option 2 (not yet done) for a
# proper trial-level trace of what's causing these.
com_mask = df["com.0"].abs() <= 200

clean = df[
    (df["is_recording"] == 1000.0) &
    (df["perturbation_mode"] == "regular") &
    task_mask &
    com_mask
].copy()

# Reconstruct trial_id from state changes (side flips = new trial).
clean["side"] = clean["current_state"].str.contains("LEFT").map({True: "left", False: "right"})
clean["trial_id"] = (clean["side"] != clean["side"].shift()).cumsum()

# Only drop the FINAL trial_id if remaining_trials==0 there -- that's the
# true end-of-session padding artifact. remaining_trials==0 also fires at
# mid-session block-transition resets, which are real trials and must be
# kept (confirmed: 5 of 6 previously dropped trials were false positives).
max_id = clean["trial_id"].max()
last_trial_remaining = clean.loc[clean["trial_id"] == max_id, "remaining_trials"].iloc[0]
n_dropped = 0
if last_trial_remaining == 0:
    clean = clean[clean["trial_id"] != max_id]
    n_dropped = 1

print(f"{subject}: {clean['trial_id'].nunique()} trials kept, {n_dropped} dropped as end-of-session padding.")
print("\nSign check:")
print("  RIGHT states mean:", clean[clean["current_state"].str.contains("RIGHT")]["com_x_human"].mean())
print("  LEFT states mean:", clean[clean["current_state"].str.contains("LEFT")]["com_x_human"].mean())
print("\ncom_x_human range:")
print(clean["com_x_human"].describe())

clean.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to {OUTPUT_PATH}")