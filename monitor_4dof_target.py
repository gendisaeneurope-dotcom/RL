import pandas as pd
from stable_baselines3.common.monitor import load_results

df = load_results("./training_logs_4dof_target/")
df["cum_timesteps"] = df["l"].cumsum()
df["episode"] = range(len(df))

# stage boundaries from your curriculum
boundaries = [100_000, 250_000, 400_000, 500_000]
for b in boundaries:
    ep_at_boundary = (df["cum_timesteps"] >= b).idxmax()
    print(f"Stage boundary at {b} timesteps -> episode {ep_at_boundary}")

# where did the reward jump actually happen?
print(df[["episode", "cum_timesteps", "r"]].iloc[2900:3100])
print(df[["episode", "cum_timesteps", "r"]].iloc[3100:3450:10])