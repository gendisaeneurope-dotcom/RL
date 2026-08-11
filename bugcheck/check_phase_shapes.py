import pandas as pd
import numpy as np



import pandas as pd
df = pd.read_csv(r"C:\Gepi10\SPACEMED\Sem4_Thesis\Thesis\Workspace\Gymnasium_RL\human_com_cleaned_subject003_v7.csv", low_memory=False)
print("Rows with |com_x_human| > 0.2:", (df["com_x_human"].abs() > 0.2).sum())
print()
print(df[df["block_idx"] == 5000.0]["com_x_human"].describe())

df = pd.read_csv(r'C:\Gepi10\SPACEMED\Sem4_Thesis\Thesis\Workspace\Gymnasium_RL\resynchronized_data_subject003.csv', low_memory=False)
df = df[df['perturbation_mode'] == 'regular']

task_mask = df['current_state'].isin(
    ['GO_TO_LEFT_CIRCLE_AFTER_TRIAL', 'GO_TO_RIGHT_CIRCLE_AFTER_TRIAL',
     'STAY_IN_LEFT_CIRCLE', 'STAY_IN_RIGHT_CIRCLE'])
df = df[task_mask].copy()

print(df.groupby('block_idx')['com.0'].agg(['mean', 'std']))

block5000 = df[df['block_idx'] == 5000.0]
print(block5000['com.0'].describe())
print()
print("Number of rows with |com.0| > 200:", (block5000['com.0'].abs() > 200).sum(), "out of", len(block5000))