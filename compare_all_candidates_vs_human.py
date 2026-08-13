"""
compare_all_candidates_vs_human.py
===================================
Human vs ALL THREE candidates, correct axis pairing, base-of-support
normalization. Generalizes compare_human_vs_c2_yaxis.py using the same
CANDIDATES-dict pattern as watch.py.

    python compare_all_candidates_vs_human.py
    python compare_all_candidates_vs_human.py --candidates candidate2 candidate3
"""
import argparse
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
HUMAN_AP_COL = "com_ap_human"
TRIAL_COL = "trial_id"
HUMAN_ML_HALFWIDTH = 0.1205 

RESAMPLE_LEN = 60
N_EPISODES = 40

# Update model/vecnorm names to whichever confirmed run you're reporting
# for each candidate. All three trained at the corrected geometry
# (0.15m base of support) and displacement (0.107m).
CANDIDATES = {
    "candidate1": dict(env_cls=Candidate1Env, model="ppo_c1_yaxis_oa015_s0",
                       vecnorm="vecnormalize_c1_yaxis_oa015_s0.pkl"),
    "candidate2": dict(env_cls=Candidate2Env, model="ppo_candidate2_yaxis_015_107",
                       vecnorm="vecnormalize_candidate2_yaxis_015_107.pkl"),
    "candidate3": dict(env_cls=Candidate3Env, model="ppo_candidate3_yaxis_050_1",
                       vecnorm="vecnormalize_candidate3_yaxis_050_1.pkl"),
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
    model = PPO.load(model_path)
    probe_env = env_cls()
    sim_ml_halfwidth = probe_env.base_half_width

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
    return sims, sim_ml_halfwidth


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def band(fig, x, mean, std, color, name):
    fig.add_trace(go.Scatter(x=x, y=mean, mode="lines", name=name, line=dict(color=color, width=3)))
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([mean + std, (mean - std)[::-1]]),
        fill="toself", fillcolor=color, opacity=0.15, line=dict(width=0), showlegend=False))


def evaluate_candidate(name, cfg, trials):
    print(f"\nRolling out {name}...")
    sims, sim_ml_halfwidth = collect_sim(cfg["env_cls"], cfg["model"], cfg["vecnorm"])

    sim_task = np.array([s["task"] for s in sims])
    sim_ap = np.array([s["ap"] for s in sims])
    hum_task = np.array([t["task"] for t in trials])
    hum_ap = np.array([t["ap"] for t in trials])

    corrs = [np.mean([corr(t["task"], s["task"]) for s in sims]) for t in trials]

    hum_task_norm = [t["task"] / HUMAN_ML_HALFWIDTH for t in trials]
    sim_task_norm = [s["task"] / sim_ml_halfwidth for s in sims]
    hum_frac = np.mean([abs(t_norm[-1]) for t_norm in hum_task_norm])
    sim_frac = np.mean([abs(s_norm[-1]) for s_norm in sim_task_norm])

    hum_disp = np.mean([abs(t['task'][-1]) for t in trials])
    sim_disp = np.mean([abs(s['task'][-1]) for s in sims])
    ap_hum = np.mean([np.ptp(t['ap']) for t in trials])
    ap_sim = np.mean([np.ptp(s['ap']) for s in sims])

    x = np.linspace(0, 1, RESAMPLE_LEN)
    fig = go.Figure()
    band(fig, x, sim_task.mean(0), sim_task.std(0), "royalblue", f"{name} (sim)")
    band(fig, x, hum_task.mean(0), hum_task.std(0), "firebrick", "human (ML)")
    fig.update_layout(title=f"TASK AXIS: human vs {name}",
                      xaxis_title="normalized time", yaxis_title="displacement from start (m)")
    fig.write_html(f"task_axis_{name}.html")

    return dict(name=name, corr=np.mean(corrs), sim_disp=sim_disp, hum_disp=hum_disp,
               sim_frac=sim_frac, hum_frac=hum_frac, ap_sim=ap_sim, ap_hum=ap_hum,
               sim_ml_halfwidth=sim_ml_halfwidth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", default=list(CANDIDATES.keys()),
                       choices=list(CANDIDATES.keys()))
    args = parser.parse_args()

    print("Loading human data...")
    trials = load_human()
    print(f"  {len(trials)} trials")

    results = []
    for name in args.candidates:
        r = evaluate_candidate(name, CANDIDATES[name], trials)
        results.append(r)
        print(f"  {name}: corr={r['corr']:.4f}  disp={r['sim_disp']:.4f}m "
              f"(human {r['hum_disp']:.4f}m)  own-range={r['sim_frac']*100:.1f}% "
              f"(human {r['hum_frac']*100:.1f}%)  AP p-p={r['ap_sim']:.4f}m "
              f"(human {r['ap_hum']:.4f}m)")

    print("\n" + "=" * 100)
    print(f"{'candidate':>12} {'corr':>7} {'sim_disp':>9} {'hum_disp':>9} "
          f"{'sim_%own':>9} {'hum_%own':>9} {'AP_sim':>8} {'AP_hum':>8}")
    print("-" * 100)
    for r in results:
        print(f"{r['name']:>12} {r['corr']:>7.4f} {r['sim_disp']:>9.4f} {r['hum_disp']:>9.4f} "
              f"{r['sim_frac']*100:>8.1f}% {r['hum_frac']*100:>8.1f}% "
              f"{r['ap_sim']:>8.4f} {r['ap_hum']:>8.4f}")
    print("=" * 100)
    print("\nHuman baseline for reference: task-axis corr ~0.83 (human-vs-human)")
    pd.DataFrame(results).to_csv("all_candidates_vs_human_summary.csv", index=False)
    print("Saved all_candidates_vs_human_summary.csv")
