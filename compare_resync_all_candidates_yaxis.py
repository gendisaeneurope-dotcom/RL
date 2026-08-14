"""
compare_resync_all_candidates_yaxis.py
========================================
Extends compare_resync_yaxis.py (previously Candidate-2-only) to all three
candidates, reusing the same rollout logic already verified working in
compare_human_vs_all_candidates_yaxis.py.

Aligns each trajectory to the moment it crosses 50% of its OWN final
displacement, then stretches the before/after halves onto a shared
normalized timeline. This removes timing offset before comparing shape.

WHY: raw correlation (~0.12-0.13, confirmed across all three candidates
in the full comparison run) compares trajectories at fixed clock time.

Usage:
    python compare_resync_all_candidates_yaxis.py
"""
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
HUMAN_TASK_COL = "com_ml_human"
TRIAL_COL = "trial_id"

EXPECTED_TARGET_DISPLACEMENT = 0.107
RESAMPLE_LEN = 60
N_EPISODES = 40

# VERIFY against your actual saved files -- same filenames confirmed
# working in tonight's compare_human_vs_all_candidates_yaxis.py run.
CANDIDATES = {
    "candidate1": dict(env_cls=Candidate1Env, model="ppo_candidate1_yaxis_015",
                       vecnorm="vecnormalize_candidate1_yaxis_015.pkl"),
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


def zero_ref(traj):
    traj = np.asarray(traj, dtype=float)
    return traj - traj[0]


def resample(traj, n=RESAMPLE_LEN):
    traj = np.asarray(traj, dtype=float)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(traj)), traj)


_fallback_count = {"n": 0}


def resync_stretch(traj, anchor_frac=0.5, label=""):
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


def load_human():
    df = pd.read_csv(HUMAN_CSV, low_memory=False)
    trials = []
    for tid, g in df.groupby(TRIAL_COL):
        vals = g[HUMAN_TASK_COL].to_numpy()
        if len(vals) < 5:
            continue
        v0 = zero_ref(vals)
        if np.sign(v0[-1] - v0[0]) > 0:
            v0 = -v0
        trials.append(dict(trial_id=tid, raw=resample(v0),
                           resync=resync_stretch(v0, label=f"human trial {tid}")))
    return trials


def collect_sim(env_cls, model_path, vecnorm_path, tag, target_displacement, n_episodes=N_EPISODES):
    model = PPO.load(model_path)
    sims = []
    for ep in range(n_episodes):
        venv = DummyVecEnv([lambda: TimeLimit(
            env_cls(target_displacement=target_displacement), max_episode_steps=1000)])
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
        venv.seed(ep)
        obs = venv.reset()
        traj = []
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, d, info = venv.step(a)
            done = bool(d[0])
            traj.append(float(info[0]["com_y"]))
        venv.close()
        v0 = zero_ref(traj)
        if np.sign(v0[-1] - v0[0]) > 0:
            v0 = -v0
        sims.append(dict(episode=ep, raw=resample(v0),
                         resync=resync_stretch(v0, label=f"{tag} episode {ep}")))
    return sims


def run_one_candidate(tag, cfg, trials):
    print(f"\n{'='*60}\nCANDIDATE: {tag}\n{'='*60}")
    print("Rolling out sim...")
    sims = collect_sim(cfg["env_cls"], cfg["model"], cfg["vecnorm"], tag, EXPECTED_TARGET_DISPLACEMENT)

    raw_corrs = [np.mean([corr(t["raw"], s["raw"]) for s in sims]) for t in trials]
    resync_corrs = [np.mean([corr(t["resync"], s["resync"]) for s in sims]) for t in trials]

    raw_mean, resync_mean = float(np.mean(raw_corrs)), float(np.mean(resync_corrs))

    print(f"\n=== [{tag}] RAW (fixed clock time) ===")
    print(f"Mean correlation: {raw_mean:.4f}")
    print(f"\n=== [{tag}] RESYNCED (aligned to 50%-displacement point) ===")
    print(f"Mean correlation: {resync_mean:.4f}")
    print(f"  Gap (resync - raw): {resync_mean - raw_mean:+.4f}")

    x = np.linspace(0, 1, RESAMPLE_LEN)
    hum_resync = np.array([t["resync"] for t in trials])
    sim_resync = np.array([s["resync"] for s in sims])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=hum_resync.mean(0), mode="lines", name="human (resynced)",
                             line=dict(color="#CC79A7", width=3)))
    fig.add_trace(go.Scatter(x=x, y=sim_resync.mean(0), mode="lines", name=f"{tag} (resynced)",
                             line=dict(color=CANDIDATE_COLORS.get(tag, "#0072B2"), width=3)))
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="resync anchor (50% displacement)")
    fig.update_layout(title=f"Resynchronized comparison: human vs {tag} (task axis)",
                      xaxis_title="resynchronized normalized time",
                      yaxis_title="displacement from start (m)",
                      template="plotly_white")
    fig.write_html(f"resync_comparison_{tag}.html")
    print(f"Saved resync_comparison_{tag}.html")

    return dict(tag=tag, raw=raw_mean, resync=resync_mean, sim_resync_mean=sim_resync.mean(0))


if __name__ == "__main__":
    print("Loading human data...")
    trials = load_human()
    print(f"  {len(trials)} trials")

    all_results = {}
    for tag, cfg in CANDIDATES.items():
        all_results[tag] = run_one_candidate(tag, cfg, trials)

    print(f"\n{'='*60}\nSUMMARY -- raw vs resync, all candidates\n{'='*60}")
    print(f"{'candidate':<12}{'raw_corr':>10}{'resync_corr':>13}{'gap':>10}")
    for tag, r in all_results.items():
        print(f"{tag:<12}{r['raw']:>10.4f}{r['resync']:>13.4f}{r['resync']-r['raw']:>10.4f}")

    if _fallback_count["n"] > 0:
        print(f"\nWARNING: {_fallback_count['n']} trial(s)/episode(s) across all runs fell back to "
              f"unaligned resample (never crossed their own 50%-displacement anchor). "
              f"Treat resync numbers with extra caution if this count is large relative to "
              f"total trials + episodes.")
    else:
        print("\nNo fallback cases -- every trial/episode crossed its own 50%-displacement anchor cleanly.")

    x = np.linspace(0, 1, RESAMPLE_LEN)
    hum_resync = np.array([t["resync"] for t in trials])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=hum_resync.mean(0), mode="lines", name="human (resynced)",
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
    print("\nSaved resync_comparison_all_candidates.html")
