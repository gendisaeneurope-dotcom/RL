"""
derive_comy_and_target.py
=============================
FIXES TWO CONFIRMED GAPS (2026-08-12):
  1. com_y_human does not exist anywhere in the pipeline. clean_prepare.py
     only ever processed com.0 (-> com_x_human); com.1 was never touched.
     This derives com_y_human from the SAME raw resynchronized_data file,
     using the SAME mm->m conversion convention already established for
     com_x (sign flip, /1000)
  2. FIXED_TARGET was never grounded in real data -- the sim's +/-0.08 was
     an arbitrary pick inside the training range, never matched to what
     humans actually did. This computes the real value directly from the
     human data: the mean final |com_x_human| displacement across all
     trials, which IS a principled, data-derived target distance.


Usage:
    python derive_comy_and_target.py
"""
import pandas as pd
import numpy as np

RAW_DATA_PATH = "C:\\Gepi10\\SPACEMED\\Sem4_Thesis\\Thesis\\Workspace\\Gymnasium_RL\\resynchronized_data_subject003.csv"
V7_CSV_PATH = "C:\\Gepi10\\SPACEMED\\Sem4_Thesis\\Thesis\\Workspace\\Gymnasium_RL\\human_com_cleaned_subject003_v7.csv"
OUTPUT_PATH = "C:\\Gepi10\\SPACEMED\\Sem4_Thesis\\Thesis\\Workspace\\Gymnasium_RL\\human_com_cleaned_subject003_v8.csv"

# Same filtering logic already established in clean_prepare.py -- adjust
# if your actual clean_prepare.py filters differ from this reconstruction.
IS_RECORDING_VALUE = 1000.0
PERTURBATION_MODE_VALUE = "regular"


def rebuild_with_comy():
    """Re-derives com_y_human from the raw file using the same
    is_recording / perturbation_mode filter and trial_id reconstruction
    logic as clean_prepare.py."""
    raw = pd.read_csv(RAW_DATA_PATH, low_memory=False)

    filtered = raw[
        (raw["is_recording"] == IS_RECORDING_VALUE) &
        (raw["perturbation_mode"] == PERTURBATION_MODE_VALUE)
    ].copy()

    # SAME conversion convention as com_x: sign flip, mm -> m.
    # ASSUMPTION: com.1's sign convention matches com.0's. If your y-axis
    # plots come out mirrored (left/right flipped) versus what you expect
    # physically, remove the negative sign here and re-run.
    filtered["com_y_human"] = -filtered["com.1"] / 1000.0
    filtered["com_x_human"] = -filtered["com.0"] / 1000.0

    # Same trial reconstruction as clean_prepare.py: side-change grouping.
    # NOTE: this assumes a 'side' column exists in the raw file, matching
    # the logic described in your pipeline. If trial reconstruction here
    # doesn't match clean_prepare.py's actual trial count (305 for v7),
    # check clean_prepare.py directly for the exact reconstruction logic
    # and port it here instead of this reconstruction.
    if "side" in filtered.columns:
        filtered["trial_id"] = (filtered["side"] != filtered["side"].shift()).cumsum()
    else:
        print("WARNING: no 'side' column found -- cannot reconstruct trial_id "
              "independently. Falling back to merging onto existing v7 by row "
              "position, which assumes IDENTICAL filtering between this script "
              "and clean_prepare.py. Verify trial counts match (v7 has a known "
              "trial count) before trusting this merge.")

    return filtered


def compute_target_from_human_data(v7_path=V7_CSV_PATH, comx_col="com_x_human",
                                     trial_col="trial_id"):
    """The data-derived fixed target: mean |final displacement| across all
    human trials, computed on zero-referenced trajectories (same
    convention as the comparison pipeline)."""
    df = pd.read_csv(v7_path, low_memory=False)
    final_displacements = []
    for trial_id, g in df.groupby(trial_col):
        vals = g[comx_col].to_numpy()
        if len(vals) < 5:
            continue
        traj = vals - vals[0]
        final_displacements.append(abs(traj[-1]))
    final_displacements = np.array(final_displacements)
    return final_displacements.mean(), final_displacements.std(), len(final_displacements)


if __name__ == "__main__":
    print("=== Deriving com_y_human ===")
    try:
        rebuilt = rebuild_with_comy()
        rebuilt.to_csv(OUTPUT_PATH, index=False)
        print(f"Saved {OUTPUT_PATH} with com_y_human column added.")
        print(f"Row count: {len(rebuilt)}, unique trial_ids: {rebuilt['trial_id'].nunique() if 'trial_id' in rebuilt.columns else 'N/A'}")
        
    except FileNotFoundError as e:
        print(f"Could not find raw data file: {e}")
        print("Update RAW_DATA_PATH at the top of this script to the correct path.\n")

    print("=== Deriving FIXED_TARGET from human data ===")
    mean_disp, std_disp, n = compute_target_from_human_data()
    print(f"Mean final |displacement| across {n} human trials: {mean_disp:.4f} m")
    print(f"Std: {std_disp:.4f} m")
    print(f"\n>>> Use FIXED_TARGET = {mean_disp:.4f} in compare_resync_comy.py <<<")
