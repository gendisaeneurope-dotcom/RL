import mujoco
m = mujoco.MjModel.from_xml_path("ankle_knee_hip_trunk.xml")
print(m.nu, m.nbody, [m.body(i).name for i in range(m.nbody)])

import pandas as pd
df = pd.read_csv("resynchronized_data_subject003.csv")
print(df.shape)
print(df.columns.tolist())
print(df.head())
print(df.describe())