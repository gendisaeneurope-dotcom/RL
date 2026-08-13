"""
compare_human_vs_all_candidates_yaxis.py
=========================================
Human vs ALL THREE candidates, correct axis pairing, PLUS base-of-support
(BoS) normalization. Generalizes compare_human_vs_c2_yaxis.py (previously
Candidate-2-only) to loop over CANDIDATES and produce per-candidate output
files, plus one combined path plot with all three candidates overlaid
against human data.


Usage:
    python compare_human_vs_all_candidates_yaxis.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from candidate1_yaxis import Candidate1Env   # TODO: confirm module/class name
from candidate2_yaxis import Candidate2Env
from candidate3_yaxis import Candidate3Env   # TODO: confirm module/class name


HUMAN_CSV = "C:\\Gepi10\\SPACEMED\\Sem4_Thesis\\Thesis\\Workspace\\Gymnasium_RL\\human_com_cleaned_subject003_v9.csv"
HUMAN_TASK_COL = "com_ml_human"    # pairs with sim com_y
HUMAN_AP_COL = "com_ap_human"      # pairs with sim com_x
TRIAL_COL = "trial_id"

# Real, measured -- see header note. NOT an estimate.
HUMAN_ML_HALFWIDTH = 0.1205

RESAMPLE_LEN = 60
N_EPISODES = 40

CANDIDATES = {
    "candidate1": dict(env_cls=Candidate1Env, model="ppo_c1_yaxis_oa015_s0",
                       vecnorm="vecnormalize_c1_yaxis_oa015_s0.pkl"),
    "candidate2": dict(env_cls=Candidate2Env, model="ppo_candidate2_yaxis_015_107",
                       vecnorm="vecnormalize_candidate2_yaxis_015_107.pkl"),
    "candidate3": dict(env_cls=Candidate3Env, model="ppo_candidate3_yaxis_050_1",
                       vecnorm="vecnormalize_candidate3_yaxis_050_1.pkl"),
}

CANDIDATE_COLORS = {
    "candidate1": "seagreen",
    "candidate2": "royalblue",
    "candidate3": "darkorange",
}


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
        if np.sign(t0[-1] - t0[0]) > 0:
            t0, a0 = -t0, -a0
        trials.append({"trial_id": tid, "task": resample(t0), "ap": resample(a0)})
    return trials


def collect_sim(env_cls, model_path, vecnorm_path, n_episodes=N_EPISODES):
    """Also returns the sim's actual ML half-width, read from the live env
    rather than hardcoded, so it can never drift out of sync with the model."""
    model = PPO.load(model_path)

    probe_env = env_cls()
    sim_ml_halfwidth = probe_env.base_half_width
    print(f"  sim ML half-width (from foot_geom): {sim_ml_halfwidth:.4f} m")

    sims = []
    for ep in range(n_episodes):
        venv = DummyVecEnv([lambda: TimeLimit(env_cls(), max_episode_steps=1000)])
        venv = VecNormalize.load(vecnorm_path, venv)
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
            task_traj.append(float(info[0]["com_y"]))
            ap_traj.append(float(info[0]["com_x"]))
        venv.close()

        t0 = zero_ref(task_traj)
        a0 = zero_ref(ap_traj)
        if np.sign(t0[-1] - t0[0]) > 0:
            t0, a0 = -t0, -a0
        sims.append({"episode": ep, "task": resample(t0), "ap": resample(a0)})
        print(f"    episode {ep}: {len(task_traj)} steps")
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


def run_one_candidate(tag, cfg, trials, hum_task, hum_ap):
    print(f"\n{'='*60}\nCANDIDATE: {tag}\n{'='*60}")
    print("Rolling out sim...")
    sims, sim_ml_halfwidth = collect_sim(cfg["env_cls"], cfg["model"], cfg["vecnorm"])

    sim_task = np.array([s["task"] for s in sims])
    sim_ap = np.array([s["ap"] for s in sims])

    # RAW comparison
    corrs, rmses = [], []
    for t in trials:
        corrs.append(np.mean([corr(t["task"], s["task"]) for s in sims]))
        rmses.append(np.mean([rmse(t["task"], s["task"]) for s in sims]))

    print(f"\n=== [{tag}] TASK AXIS, RAW (human ML vs sim com_y, metres) ===")
    print(f"Mean correlation: {np.mean(corrs):.4f}")
    print(f"Mean RMSE:        {np.mean(rmses):.4f} m")
    print(f"Human final displacement: {np.mean([abs(t['task'][-1]) for t in trials]):.4f} m")
    print(f"Sim final displacement:   {np.mean([abs(s['task'][-1]) for s in sims]):.4f} m")
    print("(human baseline for reference: corr ~0.83)")

    # NORMALIZED comparison
    hum_task_norm = [t["task"] / HUMAN_ML_HALFWIDTH for t in trials]
    sim_task_norm = [s["task"] / sim_ml_halfwidth for s in sims]

    corrs_norm, rmses_norm = [], []
    for t_norm in hum_task_norm:
        corrs_norm.append(np.mean([corr(t_norm, s_norm) for s_norm in sim_task_norm]))
        rmses_norm.append(np.mean([rmse(t_norm, s_norm) for s_norm in sim_task_norm]))

    hum_frac = np.mean([abs(t_norm[-1]) for t_norm in hum_task_norm])
    sim_frac = np.mean([abs(s_norm[-1]) for s_norm in sim_task_norm])
    frac_gap = abs(hum_frac - sim_frac)

    print(f"\n=== [{tag}] TASK AXIS, NORMALIZED (fraction of own ML base-of-support) ===")
    print(f"Human ML half-width used: {HUMAN_ML_HALFWIDTH:.4f} m (measured, toe markers, subject003)")
    print(f"Sim ML half-width used:   {sim_ml_halfwidth:.4f} m (read from foot_geom, model={cfg['model']})")
    print(f"Mean correlation: {np.mean(corrs_norm):.4f}")
    print(f"Mean RMSE:        {np.mean(rmses_norm):.4f}  (dimensionless, fraction of own BoS)")
    print(f"Human final displacement: {hum_frac:.4f}  ({hum_frac*100:.1f}% of own ML half-width)")
    print(f"Sim final displacement:   {sim_frac:.4f}  ({sim_frac*100:.1f}% of own ML half-width)")
    print(f"  --> Gap in own-range usage: {frac_gap:.4f} ({frac_gap*100:.1f} percentage points)")

    print(f"\n=== [{tag}] AP AXIS (human AP vs sim com_x) ===")
    print(f"Human peak-to-peak: {np.mean([np.ptp(t['ap']) for t in trials]):.4f} m")
    print(f"Sim peak-to-peak:   {np.mean([np.ptp(s['ap']) for s in sims]):.4f} m")

    x = np.linspace(0, 1, RESAMPLE_LEN)

    fig = go.Figure()
    band(fig, x, sim_task.mean(0), sim_task.std(0), "royalblue", f"sim (com_y) [{tag}]")
    band(fig, x, hum_task.mean(0), hum_task.std(0), "firebrick", "human (ML)")
    fig.update_layout(title=f"TASK AXIS, raw metres: human ML vs sim com_y -- {tag}",
                      xaxis_title="normalized time", yaxis_title="displacement from start (m)")
    fig.write_html(f"task_axis_{tag}.html")
    print(f"Saved task_axis_{tag}.html")

    sim_task_norm_arr = np.array(sim_task_norm)
    hum_task_norm_arr = np.array(hum_task_norm)
    fig = go.Figure()
    band(fig, x, sim_task_norm_arr.mean(0), sim_task_norm_arr.std(0), "royalblue", "sim (fraction of own BoS)")
    band(fig, x, hum_task_norm_arr.mean(0), hum_task_norm_arr.std(0), "firebrick", "human (fraction of own BoS)")
    fig.update_layout(title=f"TASK AXIS, normalized: human vs sim, fraction of own ML BoS -- {tag}",
                      xaxis_title="normalized time", yaxis_title="displacement / own ML half-width")
    fig.write_html(f"task_axis_normalized_{tag}.html")
    print(f"Saved task_axis_normalized_{tag}.html")

    fig = go.Figure()
    band(fig, x, sim_ap.mean(0), sim_ap.std(0), "royalblue", "sim (com_x)")
    band(fig, x, hum_ap.mean(0), hum_ap.std(0), "firebrick", "human (AP)")
    fig.update_layout(title=f"AP AXIS: human AP vs sim com_x -- {tag}",
                      xaxis_title="normalized time", yaxis_title="displacement from start (m)")
    fig.write_html(f"ap_axis_{tag}.html")
    print(f"Saved ap_axis_{tag}.html")

    # Per-candidate CoM path plot (task axis vs AP axis)
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
    fig.update_layout(title=f"CoM path: task axis vs AP axis -- {tag}",
                      xaxis_title="task axis displacement (m)",
                      yaxis_title="AP axis displacement (m)")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.write_html(f"path_{tag}.html")
    print(f"Saved path_{tag}.html")

    return {
        "tag": tag,
        "sim_task_mean": sim_task.mean(0), "sim_ap_mean": sim_ap.mean(0),
        "raw_corr": float(np.mean(corrs)), "raw_rmse": float(np.mean(rmses)),
        "norm_corr": float(np.mean(corrs_norm)), "norm_rmse": float(np.mean(rmses_norm)),
        "hum_frac": hum_frac, "sim_frac": sim_frac, "frac_gap": frac_gap,
        "sim_ml_halfwidth": sim_ml_halfwidth,
    }


if __name__ == "__main__":
    print("Loading human data...")
    trials = load_human()
    print(f"  {len(trials)} trials")

    hum_task = np.array([t["task"] for t in trials])
    hum_ap = np.array([t["ap"] for t in trials])

    all_results = {}
    for tag, cfg in CANDIDATES.items():
        all_results[tag] = run_one_candidate(tag, cfg, trials, hum_task, hum_ap)

    # ---- Summary table across all candidates ----
    print(f"\n{'='*60}\nSUMMARY -- all candidates vs human\n{'='*60}")
    print(f"{'candidate':<12}{'raw_corr':>10}{'raw_rmse':>10}{'norm_corr':>11}"
          f"{'norm_rmse':>11}{'hum_frac':>10}{'sim_frac':>10}{'gap':>8}")
    for tag, r in all_results.items():
        print(f"{tag:<12}{r['raw_corr']:>10.4f}{r['raw_rmse']:>10.4f}{r['norm_corr']:>11.4f}"
              f"{r['norm_rmse']:>11.4f}{r['hum_frac']:>10.4f}{r['sim_frac']:>10.4f}{r['frac_gap']:>8.4f}")

    # ---- Combined path plot: human + all three candidates ----
    x = np.linspace(0, 1, RESAMPLE_LEN)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hum_task.mean(0), y=hum_ap.mean(0), mode="lines",
                             name="human mean", line=dict(color="firebrick", width=4)))
    for tag, r in all_results.items():
        fig.add_trace(go.Scatter(x=r["sim_task_mean"], y=r["sim_ap_mean"], mode="lines",
                                 name=f"{tag} mean",
                                 line=dict(color=CANDIDATE_COLORS.get(tag, "gray"), width=3)))
    fig.update_layout(title="CoM path: task axis vs AP axis -- all candidates vs human",
                      xaxis_title="task axis displacement (m)",
                      yaxis_title="AP axis displacement (m)")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.write_html("path_all_candidates.html")
    print("\nSaved path_all_candidates.html")

    # ---- Combined task-axis overlay: human + all three candidates ----
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=hum_task.mean(0), mode="lines", name="human (ML)",
                             line=dict(color="firebrick", width=4)))
    for tag, r in all_results.items():
        fig.add_trace(go.Scatter(x=x, y=r["sim_task_mean"], mode="lines", name=f"{tag}",
                                 line=dict(color=CANDIDATE_COLORS.get(tag, "gray"), width=3)))
    fig.update_layout(title="TASK AXIS: human vs all candidates (raw metres)",
                      xaxis_title="normalized time", yaxis_title="displacement from start (m)")
    fig.write_html("task_axis_all_candidates.html")
    print("Saved task_axis_all_candidates.html")
