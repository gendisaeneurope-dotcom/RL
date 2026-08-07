"""
Human data prep v2, simply REVERSE com_x on
all rightward-moving trials so every human trial points the same canonical
direction (leftward / decreasing, matching the convention already
established when we sign-checked com.0 at the start of this analysis).


Usage:
    python prepare_human_data_v2.py
"""
import pandas as pd
import numpy as np

INPUT_PATH = "C:\\Gepi10\\SPACEMED\\Sem4_Thesis\\Thesis\\Workspace\\Gymnasium_RL\\human_com_cleaned_subject003.csv"
OUTPUT_PATH = "C:\\Gepi10\\SPACEMED\\Sem4_Thesis\\Thesis\\Workspace\\Gymnasium_RL\\human_com_cleaned_subject003_new.csv"

df = pd.read_csv(INPUT_PATH, low_memory=False)
 
trial_directions = df.groupby("trial_id")["com_x_human"].apply(
    lambda vals: np.sign(vals.to_numpy()[-1] - vals.to_numpy()[0])
)
 
df["trial_direction"] = df["trial_id"].map(trial_directions)
 
# Reverse com_x on every trial that moved rightward (positive direction),
# so ALL human trials end up pointing the same (leftward/decreasing) way.
df["com_x_aligned"] = np.where(
    df["trial_direction"] > 0,
    -df["com_x_human"],
    df["com_x_human"],
)
 
n_reversed = (df["trial_direction"] > 0).groupby(df["trial_id"]).first().sum()
n_total = df["trial_id"].nunique()
print(f"Reversed {n_reversed} of {n_total} trials (rightward -> flipped to align with leftward).")
 
check = df.groupby("trial_id").agg(
    direction=("trial_direction", "first"),
    start=("com_x_aligned", "first"),
    end=("com_x_aligned", "last"),
).reset_index()
check["net_move"] = check["end"] - check["start"]
print("\nSanity check -- ALL trials should now have net_move <= 0 (or very close to 0):")
print(f"  trials with net_move > 0.01: {(check['net_move'] > 0.01).sum()} / {len(check)}")
print(check["net_move"].describe())
 
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved aligned data to {OUTPUT_PATH}")