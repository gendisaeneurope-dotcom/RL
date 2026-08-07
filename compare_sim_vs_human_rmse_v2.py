"""
direction-matched comparison. Only compares each human trial against
sim episodes whose target is on the SAME side (left/right of center),
so opposite-direction trials don't cancel each other out in the mean
correlation. This is the fix for the near-zero mean correlation seen in
v1, which was an artifact of averaging mismatched directions together.

Usage:
    python compare_sim_vs_human_rmse_v2.py candidate1_ap
    python compare_sim_vs_human_rmse_v2.py candidate2_ap
    python compare_sim_vs_human_rmse_v2.py candidate3_ap
"""
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


HUMAN_CSV_PATH = "C:\\Gepi10\\SPACEMED\\Sem4_Thesis\\Thesis\\Workspace\\Gymnasium_RL\\RL\\human_com_cleaned_subject003_v2.csv"
HUMAN_COMX_COL = "com_x_aligned"
HUMAN_TRIAL_COL = "trial_id"
RESAMPLE_LEN = 60
N_SIM_EPISODES = 40  # increased from 20 to get better left/right coverage

CONFIGS = {
    "candidate1_ap": dict(module="candidate1_ap", cls="Candidate1Env",
                             model="ppo_candidate1_F_ap", vecnorm="vecnormalize_candidate1_F_ap.pkl",
                             log_dir="./training_logs_candidate1_F_ap/"),
    "candidate1_ap_comy1": dict(module="candidate1_ap", cls="Candidate1Env",
                                     model="ppo_candidate1_ap_comy1", vecnorm="vecnormalize_candidate1_ap_comy1.pkl",
                                     log_dir="./training_logs_candidate1_ap_comy1/"),                          
    "candidate2_ap": dict(module="candidate2_ap", cls="Candidate2Env",
                           model="ppo_candidate2_ap", vecnorm="vecnormalize_candidate2_ap.pkl",
                           log_dir="./training_logs_candidate2_ap/"),
    "candidate2_ap_comy1": dict(module="candidate2_ap", cls="Candidate2Env",
                               model="ppo_candidate2_ap_comy1", vecnorm="vecnormalize_candidate2_ap_comy1.pkl",
                               log_dir="./training_logs_candidate2_ap_comy1/"),
    "candidate3_ap": dict(module="candidate3_ap", cls="Candidate3Env",
                           model="ppo_candidate3_ap", vecnorm="vecnormalize_candidate3_ap.pkl",
                           log_dir="./training_logs_candidate3_ap/"),
    "candidate3_ap_comy1": dict(module="candidate3_ap", cls="Candidate3Env",
                                model="ppo_candidate3_ap_comy1", vecnorm="vecnormalize_candidate3_ap_comy1.pkl",
                                log_dir="./training_logs_candidate3_ap_comy1/"),
}


def resample_to_fixed_length(traj, length=RESAMPLE_LEN):
    traj = np.asarray(traj, dtype=float)
    old_x = np.linspace(0, 1, len(traj))
    new_x = np.linspace(0, 1, length)
    return np.interp(new_x, old_x, traj)


def load_human_trials():
    df = pd.read_csv(HUMAN_CSV_PATH, low_memory=False)
    trials = []
    for trial_id, g in df.groupby(HUMAN_TRIAL_COL):
        vals = g[HUMAN_COMX_COL].to_numpy()
        if len(vals) < 5:
            continue
        net_direction = np.sign(vals[-1] - vals[0])
        trials.append({
            "trial_id": trial_id,
            "traj": resample_to_fixed_length(vals),
            "direction": net_direction,
            "start": vals[0],
            "end": vals[-1],
        })
    return trials


def collect_sim_trajectories(cfg_key, n_episodes=N_SIM_EPISODES):
    cfg = CONFIGS[cfg_key]
    module = __import__(cfg["module"])
    EnvClass = getattr(module, cfg["cls"])
    model = PPO.load(cfg["model"])

    sims = []
    for ep in range(n_episodes):
        target = 0.08 if ep % 2 == 0 else -0.08
        venv = DummyVecEnv([lambda: TimeLimit(EnvClass(fixed_target=target), max_episode_steps=1000)])
        venv = VecNormalize.load(cfg["vecnorm"], venv)
        venv.training = False
        venv.norm_reward = False
        venv.seed(ep)
        obs = venv.reset()

        com_x_traj, target_x = [], None
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done_v, info = venv.step(action)
            done = bool(done_v[0])
            com_x_traj.append(float(info[0]["com_x"]))
            if target_x is None:
                target_x = float(info[0]["target_x"])
        venv.close()

        net_direction = np.sign(target_x)
        sims.append({
            "episode": ep,
            "traj": resample_to_fixed_length(com_x_traj),
            "direction": net_direction,
            "target_x": target_x,
        })
    return sims


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def compare_direction_matched(sims, trials):
    results = []
    for t in trials:
        same_dir_sims = [s for s in sims if s["direction"] == t["direction"]]
        if not same_dir_sims:
            same_dir_sims = sims  # fallback if no same-direction sim exists

        rmses = [rmse(t["traj"], s["traj"]) for s in same_dir_sims]
        corrs = [np.corrcoef(t["traj"], s["traj"])[0, 1] for s in same_dir_sims]
        best_idx = int(np.argmin(rmses))

        results.append({
            "trial_id": t["trial_id"],
            "direction": t["direction"],
            "n_same_dir_sims": len(same_dir_sims),
            "best_sim_episode": same_dir_sims[best_idx]["episode"],
            "best_rmse": rmses[best_idx],
            "best_corr": corrs[best_idx],
            "mean_rmse_same_dir": float(np.mean(rmses)),
            "mean_corr_same_dir": float(np.mean(corrs)),
        })
    return pd.DataFrame(results)


def plot_mean_comparison(sims, trials, cfg_key):
    x = np.linspace(0, 1, RESAMPLE_LEN)

    for direction, label in [(1.0, "rightward"), (-1.0, "leftward")]:
        sim_subset = np.array([s["traj"] for s in sims if s["direction"] == direction])
        human_subset = np.array([t["traj"] for t in trials if t["direction"] == direction])
        if len(sim_subset) == 0 or len(human_subset) == 0:
            continue

        sim_mean, sim_std = sim_subset.mean(axis=0), sim_subset.std(axis=0)
        human_mean, human_std = human_subset.mean(axis=0), human_subset.std(axis=0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=sim_mean, mode="lines", name="sim mean", line=dict(color="royalblue")))
        fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]),
                                  y=np.concatenate([sim_mean + sim_std, (sim_mean - sim_std)[::-1]]),
                                  fill="toself", fillcolor="royalblue", opacity=0.15, line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=human_mean, mode="lines", name="human mean", line=dict(color="firebrick")))
        fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]),
                                  y=np.concatenate([human_mean + human_std, (human_mean - human_std)[::-1]]),
                                  fill="toself", fillcolor="firebrick", opacity=0.15, line=dict(width=0), showlegend=False))
        fig.update_layout(title=f"Sim vs. human mean CoM-x, {label} trials ({cfg_key})",
                           xaxis_title="normalized time", yaxis_title="CoM-x (m)")
        fig.write_html(f"rmse_comparison_v2_{cfg_key}_{label}.html")
        print(f"Saved plot to rmse_comparison_v2_{cfg_key}_{label}.html "
              f"(n_sim={len(sim_subset)}, n_human={len(human_subset)})")


if __name__ == "__main__":
    cfg_key = sys.argv[1] if len(sys.argv) > 1 else "candidate1_F_ap"

    trials = load_human_trials()
    sims = collect_sim_trajectories(cfg_key)

    n_right = sum(1 for t in trials if t["direction"] > 0)
    n_left = sum(1 for t in trials if t["direction"] < 0)
    print(f"Human trials: {len(trials)} total ({n_right} rightward, {n_left} leftward)")

    sim_right = sum(1 for s in sims if s["direction"] > 0)
    sim_left = sum(1 for s in sims if s["direction"] < 0)
    print(f"Sim episodes: {len(sims)} total ({sim_right} rightward, {sim_left} leftward)\n")

    results = compare_direction_matched(sims, trials)

    print("=== SUMMARY (direction-matched) ===")
    print(f"Mean best-match RMSE: {results['best_rmse'].mean():.4f} m")
    print(f"Mean best-match correlation: {results['best_corr'].mean():.4f}")
    print(f"Mean same-direction correlation: {results['mean_corr_same_dir'].mean():.4f}")
    print()
    print("--- By direction ---")
    for d, label in [(1.0, "rightward"), (-1.0, "leftward")]:
        subset = results[results["direction"] == d]
        if len(subset) > 0:
            print(f"{label}: n={len(subset)}, mean_best_corr={subset['best_corr'].mean():.4f}, "
                  f"mean_best_rmse={subset['best_rmse'].mean():.4f}")

    results.to_csv(f"rmse_results_v2_{cfg_key}.csv", index=False)
    plot_mean_comparison(sims, trials, cfg_key)
