"""
compare_full_all_candidates_yaxis.py
=====================================
MERGED script combining compare_human_vs_all_candidates_yaxis.py and
compare_resync_all_candidates_yaxis.py. Runs the sim rollout ONCE per
candidate and computes every metric from both scripts off that single
rollout, instead of rolling out twice.

Produces, per candidate:
  - TASK AXIS raw (human ML vs sim com_y, metres): correlation, RMSE,
    final displacement (human vs sim)
  - TASK AXIS normalized (fraction of own base-of-support): correlation,
    RMSE, human/sim fraction used, gap in percentage points
  - AP AXIS: peak-to-peak (human vs sim)
  - RESYNC: raw correlation (fixed clock time) vs resynced correlation
    (aligned to each trajectory's own 50%-displacement crossing), plus
    the gap between them
  - Fallback-count warning if any trial/episode never crosses its own
    50%-displacement anchor during resync
  - All the same HTML plots as both original scripts (task axis, task
    axis normalized, AP axis, CoM path, resync comparison), per
    candidate plus combined "all candidates" versions

VERIFY before running: model/vecnorm filenames in CANDIDATES below match
your actual saved files on disk. This script checks file existence
up front and fails fast with a clear message if anything is missing.

Usage:
    python compare_full_all_candidates_yaxis.py
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from candidate1_yaxis import Candidate1Env
from candidate2_yaxis import Candidate2Env
from candidate3_yaxis import Candidate3Env


HUMAN_CSV = "C:\\Gepi10\\SPACEMED\\Sem4_Thesis\\Thesis\\Workspace\\Gymnasium_RL\\human_com_cleaned_subject003_v9.csv"
HUMAN_TASK_COL = "com_ml_human"    # pairs with sim com_y
HUMAN_AP_COL = "com_ap_human"      # pairs with sim com_x
TRIAL_COL = "trial_id"

# Real, measured -- toe-marker (LTOE/RTOE) separation, subject003.
# See formula_final-1.pdf Section 2.4. NOT an estimate.
HUMAN_ML_HALFWIDTH = 0.1205

# Reported-results configuration per formula_final-1.pdf Section 2.5.
# All three candidates below MUST have been trained at this value for
# the comparison to be valid; the script checks this at runtime.
EXPECTED_TARGET_DISPLACEMENT = 0.107

RESAMPLE_LEN = 60
N_EPISODES = 40
RESYNC_ANCHOR_FRAC = 0.5

# VERIFY these three entries against your actual saved files before running.
# Weights shown are the seed-confirmed final values per formula_final-1.pdf
# Section 3 (Candidate 1: w_off=0.15, Candidate 2: w_safety=0.15,
# Candidate 3: w_safety=0.50), all at the corrected geometry
# (LML=0.15, TARGET_DISPLACEMENT=0.107, 3x10^6 training steps).
CANDIDATES = {
    "candidate1": dict(
        env_cls=Candidate1Env,
        model="ppo_candidate1_yaxis_015",          # CONFIRM: replace if your actual filename differs
        vecnorm="vecnormalize_candidate1_yaxis_015.pkl",
        target_displacement=EXPECTED_TARGET_DISPLACEMENT,
    ),
    "candidate2": dict(
        env_cls=Candidate2Env,
        model="ppo_candidate2_yaxis_015_107",
        vecnorm="vecnormalize_candidate2_yaxis_015_107.pkl",
        target_displacement=EXPECTED_TARGET_DISPLACEMENT,
    ),
    "candidate3": dict(
        env_cls=Candidate3Env,
        model="ppo_candidate3_yaxis_050_1",         # CONFIRM: does "_1" mean 0.107 (target *10 truncated) or seed 1?
        vecnorm="vecnormalize_candidate3_yaxis_050_1.pkl",
        target_displacement=EXPECTED_TARGET_DISPLACEMENT,
    ),
}

CANDIDATE_COLORS = {
    "candidate1": "seagreen",
    "candidate2": "royalblue",
    "candidate3": "darkorange",
}

_fallback_count = {"n": 0}


# ---------- shared helpers ----------

def zero_ref(traj):
    traj = np.asarray(traj, dtype=float)
    return traj - traj[0]


def resample(traj, n=RESAMPLE_LEN):
    traj = np.asarray(traj, dtype=float)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(traj)), traj)


def resync_stretch(traj, anchor_frac=RESYNC_ANCHOR_FRAC, label=""):
    traj = np.asarray(traj, dtype=float)
    total_disp = traj[-1]
    if abs(total_disp) < 1e-9:
        _fallback_count["n"] += 1
        return resample(traj)
    threshold = anchor_frac * total_disp
    crossed = np.where(traj >= threshold)[0] if total_disp > 0 else np.where(traj <= threshold)[0]
    if len(crossed) == 0:
        _fallback_count["n"] += 1
        print(f"  WARNING: {label} never crossed its own 50%-displacement anchor; "
              f"falling back to unaligned resample for this trial/episode.")
        return resample(traj)
    idx = crossed[0]
    before, after = traj[:idx + 1], traj[idx:]
    before_r = (np.interp(np.linspace(0, 1, RESAMPLE_LEN // 2), np.linspace(0, 1, max(len(before), 2)), before)
                if len(before) > 1 else np.full(RESAMPLE_LEN // 2, before[0] if len(before) else 0.0))
    after_r = (np.interp(np.linspace(0, 1, RESAMPLE_LEN - RESAMPLE_LEN // 2), np.linspace(0, 1, max(len(after), 2)), after)
               if len(after) > 1 else np.full(RESAMPLE_LEN - RESAMPLE_LEN // 2, after[0] if len(after) else 0.0))
    return np.concatenate([before_r, after_r])


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


def check_files_exist(tag, cfg):
    missing = [p for p in (cfg["model"] + ".zip", cfg["vecnorm"]) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"[{tag}] Missing file(s) before rollout: {missing}. "
            f"Verify the model/vecnorm filenames in CANDIDATES against your actual "
            f"saved output before running the full comparison."
        )


# ---------- data loading ----------

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
        trials.append({
            "trial_id": tid,
            "task": resample(t0),
            "ap": resample(a0),
            "task_resync": resync_stretch(t0, label=f"human trial {tid}"),
        })
    return trials


def collect_sim(env_cls, model_path, vecnorm_path, target_displacement, n_episodes=N_EPISODES):
    """Single rollout per episode; computes raw, normalized, AP, and resync
    trajectories all from the same run."""
    model = PPO.load(model_path)

    probe_env = env_cls(target_displacement=target_displacement)
    sim_ml_halfwidth = probe_env.base_half_width
    actual_disp = getattr(probe_env, "target_displacement", None)
    print(f"  sim ML half-width (from foot_geom): {sim_ml_halfwidth:.4f} m")
    print(f"  sim target_displacement (as configured): {actual_disp}")
    if actual_disp is not None and abs(actual_disp - EXPECTED_TARGET_DISPLACEMENT) > 1e-6:
        print(f"  WARNING: configured target_displacement ({actual_disp}) does not match "
              f"EXPECTED_TARGET_DISPLACEMENT ({EXPECTED_TARGET_DISPLACEMENT}). "
              f"This candidate may not be comparable to the others -- verify before trusting results.")

    sims = []
    for ep in range(n_episodes):
        venv = DummyVecEnv([lambda: TimeLimit(
            env_cls(target_displacement=target_displacement), max_episode_steps=1000)])
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
        sims.append({
            "episode": ep,
            "task": resample(t0),
            "ap": resample(a0),
            "task_resync": resync_stretch(t0, label=f"{tag_for_log} episode {ep}"),
        })
        print(f"    episode {ep}: {len(task_traj)} steps")
    return sims, sim_ml_halfwidth


# ---------- per-candidate analysis ----------

def run_one_candidate(tag, cfg, trials, hum_task, hum_ap):
    global tag_for_log
    tag_for_log = tag
    print(f"\n{'='*60}\nCANDIDATE: {tag}\n{'='*60}")
    check_files_exist(tag, cfg)
    print("Rolling out sim...")
    sims, sim_ml_halfwidth = collect_sim(
        cfg["env_cls"], cfg["model"], cfg["vecnorm"], cfg["target_displacement"])

    sim_task = np.array([s["task"] for s in sims])
    sim_ap = np.array([s["ap"] for s in sims])

    # --- RAW task axis (fixed clock time) ---
    corrs, rmses = [], []
    for t in trials:
        corrs.append(np.mean([corr(t["task"], s["task"]) for s in sims]))
        rmses.append(np.mean([rmse(t["task"], s["task"]) for s in sims]))
    raw_corr_mean = float(np.mean(corrs))
    raw_rmse_mean = float(np.mean(rmses))

    print(f"\n=== [{tag}] TASK AXIS, RAW (human ML vs sim com_y, metres) ===")
    print(f"Mean correlation: {raw_corr_mean:.4f}")
    print(f"Mean RMSE:        {raw_rmse_mean:.4f} m")
    print(f"Human final displacement: {np.mean([abs(t['task'][-1]) for t in trials]):.4f} m")
    print(f"Sim final displacement:   {np.mean([abs(s['task'][-1]) for s in sims]):.4f} m")
    print("(human baseline for reference: corr ~0.82, per 3-participant baseline)")

    # --- NORMALIZED task axis (fraction of own base-of-support) ---
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

    # --- AP axis ---
    hum_ap_ptp = np.mean([np.ptp(t["ap"]) for t in trials])
    sim_ap_ptp = np.mean([np.ptp(s["ap"]) for s in sims])
    print(f"\n=== [{tag}] AP AXIS (human AP vs sim com_x) ===")
    print(f"Human peak-to-peak: {hum_ap_ptp:.4f} m")
    print(f"Sim peak-to-peak:   {sim_ap_ptp:.4f} m")

    # --- RESYNC (timing-corrected shape comparison) ---
    resync_corrs = [np.mean([corr(t["task_resync"], s["task_resync"]) for s in sims]) for t in trials]
    resync_corr_mean = float(np.mean(resync_corrs))
    resync_gap = resync_corr_mean - raw_corr_mean

    print(f"\n=== [{tag}] RESYNC (aligned to each trajectory's own 50%-displacement point) ===")
    print(f"Raw correlation (fixed clock time):  {raw_corr_mean:.4f}")
    print(f"Resynced correlation (shape only):    {resync_corr_mean:.4f}")
    print(f"Gap (resync - raw):                   {resync_gap:+.4f}")

    x = np.linspace(0, 1, RESAMPLE_LEN)

    # plot: task axis raw
    fig = go.Figure()
    band(fig, x, sim_task.mean(0), sim_task.std(0), "royalblue", f"sim (com_y) [{tag}]")
    band(fig, x, hum_task.mean(0), hum_task.std(0), "firebrick", "human (ML)")
    fig.update_layout(title=f"TASK AXIS, raw metres: human ML vs sim com_y -- {tag}",
                      xaxis_title="normalized time", yaxis_title="displacement from start (m)")
    fig.write_html(f"task_axis_{tag}.html")
    print(f"Saved task_axis_{tag}.html")

    # plot: task axis normalized
    sim_task_norm_arr = np.array(sim_task_norm)
    hum_task_norm_arr = np.array(hum_task_norm)
    fig = go.Figure()
    band(fig, x, sim_task_norm_arr.mean(0), sim_task_norm_arr.std(0), "royalblue", "sim (fraction of own BoS)")
    band(fig, x, hum_task_norm_arr.mean(0), hum_task_norm_arr.std(0), "firebrick", "human (fraction of own BoS)")
    fig.update_layout(title=f"TASK AXIS, normalized: human vs sim, fraction of own ML BoS -- {tag}",
                      xaxis_title="normalized time", yaxis_title="displacement / own ML half-width")
    fig.write_html(f"task_axis_normalized_{tag}.html")
    print(f"Saved task_axis_normalized_{tag}.html")

    # plot: AP axis
    fig = go.Figure()
    band(fig, x, sim_ap.mean(0), sim_ap.std(0), "royalblue", "sim (com_x)")
    band(fig, x, hum_ap.mean(0), hum_ap.std(0), "firebrick", "human (AP)")
    fig.update_layout(title=f"AP AXIS: human AP vs sim com_x -- {tag}",
                      xaxis_title="normalized time", yaxis_title="displacement from start (m)")
    fig.write_html(f"ap_axis_{tag}.html")
    print(f"Saved ap_axis_{tag}.html")

    # plot: CoM path
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

    # plot: resync comparison
    sim_resync_arr = np.array([s["task_resync"] for s in sims])
    hum_resync_arr = np.array([t["task_resync"] for t in trials])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=hum_resync_arr.mean(0), mode="lines", name="human (resynced)",
                             line=dict(color="#CC79A7", width=3)))
    fig.add_trace(go.Scatter(x=x, y=sim_resync_arr.mean(0), mode="lines", name=f"{tag} (resynced)",
                             line=dict(color=CANDIDATE_COLORS.get(tag, "#0072B2"), width=3)))
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="resync anchor (50% displacement)")
    fig.update_layout(title=f"Resynchronized comparison: human vs {tag} (task axis)",
                      xaxis_title="resynchronized normalized time",
                      yaxis_title="displacement from start (m)",
                      template="plotly_white")
    fig.write_html(f"resync_comparison_{tag}.html")
    print(f"Saved resync_comparison_{tag}.html")

    return {
        "tag": tag,
        "sim_task_mean": sim_task.mean(0), "sim_ap_mean": sim_ap.mean(0),
        "sim_resync_mean": sim_resync_arr.mean(0),
        "raw_corr": raw_corr_mean, "raw_rmse": raw_rmse_mean,
        "norm_corr": float(np.mean(corrs_norm)), "norm_rmse": float(np.mean(rmses_norm)),
        "hum_frac": hum_frac, "sim_frac": sim_frac, "frac_gap": frac_gap,
        "hum_ap_ptp": hum_ap_ptp, "sim_ap_ptp": sim_ap_ptp,
        "resync_corr": resync_corr_mean, "resync_gap": resync_gap,
        "sim_ml_halfwidth": sim_ml_halfwidth,
    }


tag_for_log = ""


if __name__ == "__main__":
    print("Pre-flight check: verifying all model/vecnorm files exist before any rollout...")
    for tag, cfg in CANDIDATES.items():
        check_files_exist(tag, cfg)
    print("All files found.\n")

    print("Loading human data...")
    trials = load_human()
    print(f"  {len(trials)} trials")

    hum_task = np.array([t["task"] for t in trials])
    hum_ap = np.array([t["ap"] for t in trials])

    all_results = {}
    for tag, cfg in CANDIDATES.items():
        all_results[tag] = run_one_candidate(tag, cfg, trials, hum_task, hum_ap)

    print(f"\n{'='*60}\nSUMMARY -- all candidates vs human (full metrics)\n{'='*60}")
    print(f"{'candidate':<12}{'raw_corr':>10}{'raw_rmse':>10}{'norm_corr':>11}"
          f"{'norm_rmse':>11}{'hum_frac':>10}{'sim_frac':>10}{'gap':>8}"
          f"{'hum_ap':>9}{'sim_ap':>9}{'resync':>9}{'rsync_gap':>11}")
    for tag, r in all_results.items():
        print(f"{tag:<12}{r['raw_corr']:>10.4f}{r['raw_rmse']:>10.4f}{r['norm_corr']:>11.4f}"
              f"{r['norm_rmse']:>11.4f}{r['hum_frac']:>10.4f}{r['sim_frac']:>10.4f}{r['frac_gap']:>8.4f}"
              f"{r['hum_ap_ptp']:>9.4f}{r['sim_ap_ptp']:>9.4f}{r['resync_corr']:>9.4f}{r['resync_gap']:>+11.4f}")

    if _fallback_count["n"] > 0:
        print(f"\nWARNING: {_fallback_count['n']} trial(s)/episode(s) across all runs fell back to "
              f"unaligned resample (never crossed their own 50%-displacement anchor). "
              f"Treat resync numbers with extra caution if this count is large relative to "
              f"total trials + episodes.")
    else:
        print("\nNo fallback cases -- every trial/episode crossed its own 50%-displacement anchor cleanly.")

    x = np.linspace(0, 1, RESAMPLE_LEN)

    # combined CoM path plot
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

    # combined task axis plot
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

    # combined resync plot
    hum_resync_arr = np.array([t["task_resync"] for t in trials])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=hum_resync_arr.mean(0), mode="lines", name="human (resynced)",
                             line=dict(color="#CC79A7", width=4)))
    for tag, r in all_results.items():
        fig.add_trace(go.Scatter(x=x, y=r["sim_resync_mean"], mode="lines", name=f"{tag} (resynced)",
                                 line=dict(color=CANDIDATE_COLORS.get(tag, "gray"), width=3)))
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="resync anchor (50% displacement)")
    fig.update_layout(title="Resynchronized comparison: human vs all candidates (task axis)",
                      xaxis_title="resynchronized normalized time",
                      yaxis_title="displacement from start (m)",
                      template="plotly_white")
    fig.write_html("resync_comparison_all_candidates.html")
    print("Saved resync_comparison_all_candidates.html")
