import pandas as pd
import numpy as np

df = pd.read_csv("resynchronized_data_subject003.csv")

# 1. Check CoM columns: does com differ from com_approx?
com_diff = df[["com.0","com.1","com.2"]].values - df[["com_approx.0","com_approx.1","com_approx.2"]].values
print("CoM vs CoM_approx mean abs diff (x,y,z):", np.nanmean(np.abs(com_diff), axis=0))

# 2. Target/task structure: does main_circle_position change across trials?
print("\nmain_circle_position unique count:", df[["main_circle_position.0","main_circle_position.1"]].drop_duplicates().shape[0])
print(df[["main_circle_position.0","main_circle_position.1"]].drop_duplicates().head(10))

# 3. cbos (center of base of support) - fixed or varying?
print("\ncbos unique count:", df[["cbos.0","cbos.1","cbos.2"]].drop_duplicates().shape[0])
print("cbos_set unique values:", df["cbos_set"].unique())

# 4. Joint angle ranges (deg), matching your 4-DOF model: ankle, knee, hip, trunk/lumbar
joint_cols = ["ankle_angle_r","ankle_angle_l","knee_angle_r","knee_angle_l",
              "hip_flexion_r","hip_flexion_l","lumbar_extension"]
print("\nJoint angle ranges (deg):")
print(df[joint_cols].agg(["min","max","mean","std"]).T)

# 5. Convert your simulation's ANGLE_LIMIT (0.4 rad) to degrees for direct comparison
angle_limit_deg = 0.4 * 180 / np.pi
print(f"\nYour sim ANGLE_LIMIT = 0.4 rad = {angle_limit_deg:.1f} deg")

# 6. CoM position range
print("\nCoM position range (x,y,z):")
print(df[["com.0","com.1","com.2"]].agg(["min","max","mean","std"]))

# 7. Trial/task structure
print("\nblock_idx unique:", df["block_idx"].nunique(), "| total_trials:", df["total_trials"].max())
print("current_state unique values:", df["current_state"].unique())