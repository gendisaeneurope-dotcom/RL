import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path, low_memory=False)

print(f"t_rel max (end of whole recording): {df['t_rel'].max():.2f}")
print(f"t_rel min (start of whole recording): {df['t_rel'].min():.2f}")
print(f"trial_group 308 ended at: 1639.85")
print(f"Gap between recording end and trial_group 308's end: {df['t_rel'].max() - 1639.85:.2f} seconds")

df = pd.read_csv("C:\\Gepi10\\SPACEMED\\Sem4_Thesis\\Thesis\\Workspace\\Gymnasium_RL\\human_com_cleaned_subject003.csv", low_memory=False)
print(df["trial_id"].nunique())
print(df.groupby("trial_id").size().sort_values(ascending=False).head(5))


path = r"C:\Gepi10\SPACEMED\Sem4_Thesis\Thesis\Workspace\Gymnasium_RL\resynchronized_data_subject003.csv"
df = pd.read_csv(path, low_memory=False)
print("trial_id" in df.columns)