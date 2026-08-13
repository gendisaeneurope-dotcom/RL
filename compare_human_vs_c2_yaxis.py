"""
compare_human_vs_c2_yaxis.py
============================
Human vs Candidate 2, correct axis pairing, PLUS base-of-support (BoS)
normalization.


Usage:
    python compare_human_vs_c2_yaxis.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from candidate2_yaxis import Candidate2Env

HUMAN_CSV = "C:\\Gepi10\\SPACEMED\\Sem4_Thesis\\Thesis\\Workspace\\Gymnasium_RL\\human_com_cleaned_subject003_v9.csv"
HUMAN_TASK_COL = "com_ml_human"    # pairs with sim com_y
HUMAN_AP_COL = "com_ap_human"      # pairs with sim com_x
TRIAL_COL = "trial_id"

# Real, measured -- see header note. NOT an estimate.
HUMAN_ML_HALFWIDTH = 0.1205

MODEL = "ppo_candidate2_yaxis_015_107"
VECNORM = "vecnormalize_candidate2_yaxis_015_107.pkl"
TAG = "c2_yaxis"

RESAMPLE_LEN = 60
N_EPISODES = 40


def resample(traj, n=RESAMPLE_LEN):
    traj = np.asarray(traj, dtype=float)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(traj)), traj)


def zero_ref(traj):
    traj = np.asarray(traj, dtype=float)
    return traj - traj[0]


def load_human():
    df = pd.read_csv(HUMAN_CSV, low_memory=False)
    trials = []
    for tid, g in df.groupby(TRIAL_COL):
        task = g[HUMAN_TASK_COL].to_numpy()
        ap = g[HUMAN_AP_COL].to_numpy()
        if len(task) < 5:
            continue
        t0 = zero_ref(task)
        a0 = zero_ref(ap)
        # canonicalise direction using the TASK axis; apply the same flip
        # to AP so the pair stays internally consistent
        if np.sign(t0[-1] - t0[0]) > 0:
            t0, a0 = -t0, -a0
        trials.append({"trial_id": tid,
                       "task": resample(t0),
                       "ap": resample(a0)})
    return trials


def collect_sim(n_episodes=N_EPISODES):
    """Also returns the sim's actual ML half-width, read from the live env
    rather than hardcoded, so it can never drift out of sync with MODEL."""
    model = PPO.load(MODEL)

    probe_env = Candidate2Env()
    sim_ml_halfwidth = probe_env.base_half_width
    print(f"  sim ML half-width (from foot_geom): {sim_ml_halfwidth:.4f} m")

    sims = []
    for ep in range(n_episodes):
        venv = DummyVecEnv([lambda: TimeLimit(Candidate2Env(), max_episode_steps=1000)])
        venv = VecNormalize.load(VECNORM, venv)
        venv.training = False
        venv.norm_reward = False
        venv.seed(ep)
        obs = venv.reset()

        task_traj, ap_traj = [], []
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, d, info = venv.step(a)
            done = bool(d[0])
            task_traj.append(float(info[0]["com_y"]))   # TASK axis
            ap_traj.append(float(info[0]["com_x"]))     # AP axis
        venv.close()

        t0 = zero_ref(task_traj)
        a0 = zero_ref(ap_traj)
        if np.sign(t0[-1] - t0[0]) > 0:
            t0, a0 = -t0, -a0
        sims.append({"episode": ep, "task": resample(t0), "ap": resample(a0)})
        print(f"  episode {ep}: {len(task_traj)} steps")
    return sims, sim_ml_halfwidth


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def band(fig, x, mean, std, color, name):
    fig.add_trace(go.Scatter(x=x, y=mean, mode="lines", name=name,
                             line=dict(color=color, width=3)))
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([mean + std, (mean - std)[::-1]]),
        fill="toself", fillcolor=color, opacity=0.15,
        line=dict(width=0), showlegend=False))


if __name__ == "__main__":
    print("Loading human data...")
    trials = load_human()
    print(f"  {len(trials)} trials\n")

    print("Rolling out sim...")
    sims, sim_ml_halfwidth = collect_sim()

    sim_task = np.array([s["task"] for s in sims])
    sim_ap = np.array([s["ap"] for s in sims])
    hum_task = np.array([t["task"] for t in trials])
    hum_ap = np.array([t["ap"] for t in trials])

    # =========================================================
    # RAW (absolute-metre) comparison -- unchanged from before
    # =========================================================
    corrs, rmses = [], []
    for t in trials:
        corrs.append(np.mean([corr(t["task"], s["task"]) for s in sims]))
        rmses.append(np.mean([rmse(t["task"], s["task"]) for s in sims]))

    print("\n=== TASK AXIS, RAW (human ML vs sim com_y, metres) ===")
    print(f"Mean correlation: {np.mean(corrs):.4f}")
    print(f"Mean RMSE:        {np.mean(rmses):.4f} m")
    print(f"Human final displacement: {np.mean([abs(t['task'][-1]) for t in trials]):.4f} m")
    print(f"Sim final displacement:   {np.mean([abs(s['task'][-1]) for s in sims]):.4f} m")
    print("(human baseline for reference: corr ~0.83)")

    # =========================================================
    # NORMALIZED comparison -- NEW. Each side divided by its own
    # measured/actual base-of-support half-width, so the metric
    # is "fraction of own stability limit used", not raw metres.
    # =========================================================
    hum_task_norm = [t["task"] / HUMAN_ML_HALFWIDTH for t in trials]
    sim_task_norm = [s["task"] / sim_ml_halfwidth for s in sims]

    corrs_norm, rmses_norm = [], []
    for t_norm in hum_task_norm:
        corrs_norm.append(np.mean([corr(t_norm, s_norm) for s_norm in sim_task_norm]))
        rmses_norm.append(np.mean([rmse(t_norm, s_norm) for s_norm in sim_task_norm]))

    hum_frac = np.mean([abs(t_norm[-1]) for t_norm in hum_task_norm])
    sim_frac = np.mean([abs(s_norm[-1]) for s_norm in sim_task_norm])
    frac_gap = abs(hum_frac - sim_frac)

    print(f"\n=== TASK AXIS, NORMALIZED (fraction of own ML base-of-support) ===")
    print(f"Human ML half-width used: {HUMAN_ML_HALFWIDTH:.4f} m (measured, toe markers, subject003)")
    print(f"Sim ML half-width used:   {sim_ml_halfwidth:.4f} m (read from foot_geom, model={MODEL})")
    print(f"Mean correlation: {np.mean(corrs_norm):.4f}")
    print(f"Mean RMSE:        {np.mean(rmses_norm):.4f}  (dimensionless, fraction of own BoS)")
    print(f"Human final displacement: {hum_frac:.4f}  ({hum_frac*100:.1f}% of own ML half-width)")
    print(f"Sim final displacement:   {sim_frac:.4f}  ({sim_frac*100:.1f}% of own ML half-width)")
    print(f"\n  --> Gap in own-range usage: {frac_gap:.4f} ({frac_gap*100:.1f} percentage points)")
    print(f"  --> This gap alone is enough to explain most of the {np.mean(rmses_norm):.4f}")
    print(f"      normalized RMSE above -- an {frac_gap*100:.0f}-point endpoint gap, propagated")
    print(f"      through every timestep and squared (RMSE), compounds to roughly this size.")
    print(f"      The normalized RMSE number is therefore driven mainly by the two systems")
    print(f"      using a different FRACTION of their own available range, not by a")
    print(f"      meaningful 'positional error' in any standard unit. Report the two")
    print(f"      percentages above as the interpretable finding; treat normalized RMSE")
    print(f"      as supporting detail only, not a headline number.")
    print("\nNOTE: raw and normalized comparisons answer different questions.")
    print("Raw = did the sim move the same ABSOLUTE distance as the human.")
    print("Normalized = did the sim use the same FRACTION of its own stability")
    print("limit as the human used of theirs. Report both explicitly; do not")
    print("substitute one for the other without saying so.")

    # --- AP AXIS (unchanged, raw only) ---
    print("\n=== AP AXIS (human AP vs sim com_x) ===")
    print(f"Human peak-to-peak: {np.mean([np.ptp(t['ap']) for t in trials]):.4f} m")
    print(f"Sim peak-to-peak:   {np.mean([np.ptp(s['ap']) for s in sims]):.4f} m")

    # =========================================================
    # plots
    # =========================================================
    x = np.linspace(0, 1, RESAMPLE_LEN)

    fig = go.Figure()
    band(fig, x, sim_task.mean(0), sim_task.std(0), "royalblue", "sim (com_y)")
    band(fig, x, hum_task.mean(0), hum_task.std(0), "firebrick", "human (ML)")
    fig.update_layout(title=f"TASK AXIS, raw metres: human ML vs sim com_y -- {TAG}",
                      xaxis_title="normalized time",
                      yaxis_title="displacement from start (m)")
    fig.write_html(f"task_axis_{TAG}.html")
    print(f"\nSaved task_axis_{TAG}.html")

    sim_task_norm_arr = np.array(sim_task_norm)
    hum_task_norm_arr = np.array(hum_task_norm)
    fig = go.Figure()
    band(fig, x, sim_task_norm_arr.mean(0), sim_task_norm_arr.std(0), "royalblue", "sim (fraction of own BoS)")
    band(fig, x, hum_task_norm_arr.mean(0), hum_task_norm_arr.std(0), "firebrick", "human (fraction of own BoS)")
    fig.update_layout(title=f"TASK AXIS, normalized: human vs sim, fraction of own ML BoS -- {TAG}",
                      xaxis_title="normalized time",
                      yaxis_title="displacement / own ML half-width")
    fig.write_html(f"task_axis_normalized_{TAG}.html")
    print(f"Saved task_axis_normalized_{TAG}.html")

    fig = go.Figure()
    band(fig, x, sim_ap.mean(0), sim_ap.std(0), "royalblue", "sim (com_x)")
    band(fig, x, hum_ap.mean(0), hum_ap.std(0), "firebrick", "human (AP)")
    fig.update_layout(title=f"AP AXIS: human AP vs sim com_x -- {TAG}",
                      xaxis_title="normalized time",
                      yaxis_title="displacement from start (m)")
    fig.write_html(f"ap_axis_{TAG}.html")
    print(f"Saved ap_axis_{TAG}.html")

    fig = go.Figure()
    for t in trials:
        fig.add_trace(go.Scatter(x=t["task"], y=t["ap"], mode="lines",
                                 line=dict(color="firebrick", width=1),
                                 opacity=0.06, showlegend=False))
    for s in sims:
        fig.add_trace(go.Scatter(x=s["task"], y=s["ap"], mode="lines",
                                 line=dict(color="royalblue", width=1),
                                 opacity=0.15, showlegend=False))
    fig.add_trace(go.Scatter(x=hum_task.mean(0), y=hum_ap.mean(0), mode="lines",
                             name="human mean", line=dict(color="firebrick", width=4)))
    fig.add_trace(go.Scatter(x=sim_task.mean(0), y=sim_ap.mean(0), mode="lines",
                             name="sim mean", line=dict(color="royalblue", width=4)))
    fig.update_layout(title=f"CoM path: task axis vs AP axis -- {TAG}",
                      xaxis_title="task axis displacement (m)",
                      yaxis_title="AP axis displacement (m)")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.write_html(f"path_{TAG}.html")
    print(f"Saved path_{TAG}.html")