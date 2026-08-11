import pandas as pd
import glob

log_files = glob.glob("./training_logs_candidate2_ap_comy1_delayed/monitor_*.monitor.csv")
for f in log_files[:1]:  # just check one env's log
    df = pd.read_csv(f, skiprows=1)
    print(f"n episodes logged: {len(df)}")
    print(f"mean reward (last 100 episodes): {df['r'].tail(100).mean():.2f}")
    print(f"mean reward (first 100 episodes): {df['r'].head(100).mean():.2f}")