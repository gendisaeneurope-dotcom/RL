import numpy as np
from candidate1_yaxis import Candidate1Env

env = Candidate1Env()
for ep in range(3):
    obs, info = env.reset(seed=ep)
    print(f"ep {ep}: target_y={env.target_y:+.4f}")
    print(f"  obs[8]  (com_y/S)      = {obs[8]:+.4f}  expect {env._com_xy()[1]/0.5:+.4f}")
    print(f"  obs[9]  (target_y/S)   = {obs[9]:+.4f}  expect {env.target_y/0.5:+.4f}")
    print(f"  obs[10] (error/S)      = {obs[10]:+.4f}")

import pandas as pd
import numpy as np

path = r"C:\Gepi10\SPACEMED\Sem4_Thesis\Thesis\Workspace\Gymnasium_RL\resynchronized_data_subject003.csv"
df = pd.read_csv(path, low_memory=False)

# ML axis is com.0 in this dataset's convention -- check which coordinate
# index (.0/.1/.2) corresponds to ML for the foot markers too, same way
# you verified it for CoM. Try .0 first (matches com.0 = ML) but confirm.

left_heel_ml = df["Left_foot_heel.0"]
right_heel_ml = df["Right_foot_heel.0"]

stance_width = (right_heel_ml - left_heel_ml).abs()

print("Stance width (heel-to-heel, ML axis):")
print(stance_width.describe())

# Also check toe markers as a cross-check
left_toe_ml = df["LTOE.X"]
right_toe_ml = df["RTOE.X"]
toe_width = (right_toe_ml - left_toe_ml).abs()
print("\nStance width (toe-to-toe):")
print(toe_width.describe())
