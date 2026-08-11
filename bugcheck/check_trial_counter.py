"""Checks whether trial_group 227 and 8 contain more than one value of a
trial-counter column (remaining_trials or block_idx). If either column
shows MORE THAN ONE unique value inside a trial_group, that trial_group is
actually two real trials merged together by the grouping logic.

  python check_trial_counter.py "C:\Gepi10\SPACEMED\Sem4_Thesis\Thesis\Workspace\Gymnasium_RL\resynchronized_data_subject003.csv"
"""
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path, low_memory=False)

state = df["current_state"].astype(str)
task_mask = state.isin(["GO_TO_LEFT_CIRCLE_AFTER_TRIAL", "GO_TO_RIGHT_CIRCLE_AFTER_TRIAL",
                        "STAY_IN_LEFT_CIRCLE", "STAY_IN_RIGHT_CIRCLE"])
task_df = df[task_mask].copy()
task_df["side"] = task_df["current_state"].str.contains("LEFT").map({True: "left", False: "right"})
task_df["trial_group"] = (task_df["side"] != task_df["side"].shift()).cumsum()

print("Columns available in this file:")
print(list(df.columns))
print()

for tg in [308]:
    g = task_df[task_df.trial_group == tg]
    print(f"=== trial_group {tg} ===")
    for col in ["remaining_trials", "block_idx", "trial_number", "trial_idx", "trial_count"]:
        if col in g.columns:
            uniq = g[col].unique()
            print(f"  {col}: {len(uniq)} unique value(s) -> {uniq}")
    print()
