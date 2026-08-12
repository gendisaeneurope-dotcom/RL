"""
Y-axis (mediolateral) and x/y-plane comparison: sim vs human.


Produces TWO figures:
  1. com_y vs normalized time  -- human mean+/-std against sim mean+/-std
  2. com_x vs com_y path plot  -- the 2D CoM trajectory in the horizontal
     plane, which is the supervisor's stated alternative and sidesteps any
     axis-naming ambiguity.



Usage:
    python compare_sim_vs_human_comy.py candidate2_ap_ascale1
"""
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

HUMAN_CSV_PATH = "C:\\Gepi10\\SPACEMED\\Sem4_Thesis\\Thesis\\Workspace\\Gymnasium_RL\\human_com_cleaned_subject003_v8.csv"
HUMAN_COMX_COL = "com_x_human"
HUMAN_COMY_COL = "com_y_human"
HUMAN_TRIAL_COL = "trial_id"
RESAMPLE_LEN = 60
N_SIM_EPISODES = 40

# Supervisor feedback point 3: "humans are reaching always the same target,
# you should evaluate your policy always on the same target to make it
# comparable (eventually a bit of noise in the starting position)."
# Set to a single fixed value rather than alternating +/-0.08.
FIXED_TARGET = 0.08

CONFIGS = {
    "candidate2_ap_ascale1": dict(module="candidate2_ap_comy1_staypenalty",
                                  cls="Candidate2Env",
                                  model="ppo_candidate2_ap_ascale1",
                                  vecnorm="vecnormalize_candidate2_ap_ascale1.pkl"),
}


def resample_to_fixed_length(traj, length=RESAMPLE_LEN):
    traj = np.asarray(traj, dtype=float)
    old_x = np.linspace(0, 1, len(traj))
    new_x = np.linspace(0, 1, length)
    return np.interp(new_x, old_x, traj)


def zero_reference(traj):
    traj = np.asarray(traj, dtype=float)
    return traj - traj[0]


def load_human_trials():
    """Loads BOTH x and y per trial. Direction canonicalisation is applied
    based on the x-axis net movement (the task axis, where trials alternate
    left/right); the same flip is then applied to y so that x and y stay
    consistent within a trial. If you decide y should NOT be flipped with
    x, set FLIP_Y_WITH_X = False below -- this is a genuine methodological
    choice, not an obvious default."""
    FLIP_Y_WITH_X = True

    df = pd.read_csv(HUMAN_CSV_PATH, low_memory=False)
    trials = []
    for trial_id, g in df.groupby(HUMAN_TRIAL_COL):
        vx = g[HUMAN_COMX_COL].to_numpy()
        vy = g[HUMAN_COMY_COL].to_numpy()
        if len(vx) < 5:
            continue
        vx0 = zero_reference(vx)
        vy0 = zero_reference(vy)
        net_direction = np.sign(vx0[-1] - vx0[0])
        if net_direction > 0:
            vx0 = -vx0
            if FLIP_Y_WITH_X:
                vy0 = -vy0
        trials.append({
            "trial_id": trial_id,
            "traj_x": resample_to_fixed_length(vx0),
            "traj_y": resample_to_fixed_length(vy0),
        })
    return trials


def collect_sim_trajectories(cfg_key, n_episodes=N_SIM_EPISODES):
    """Collects BOTH com_x and com_y. Uses a single FIXED_TARGET for every
    episode (supervisor feedback point 3) rather than alternating targets."""
    cfg = CONFIGS[cfg_key]
    module = __import__(cfg["module"])
    EnvClass = getattr(module, cfg["cls"])
    model = PPO.load(cfg["model"])

    sims = []
    for ep in range(n_episodes):
        venv = DummyVecEnv([lambda: TimeLimit(
            EnvClass(fixed_target=FIXED_TARGET), max_episode_steps=1000)])
        venv = VecNormalize.load(cfg["vecnorm"], venv)
        venv.training = False
        venv.norm_reward = False
        venv.seed(ep)
        obs = venv.reset()

        com_x_traj, com_y_traj = [], []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done_v, info = venv.step(action)
            done = bool(done_v[0])
            com_x_traj.append(float(info[0]["com_x"]))
            com_y_traj.append(float(info[0]["com_y"]))
        venv.close()

        x0 = zero_reference(com_x_traj)
        y0 = zero_reference(com_y_traj)
        net_direction = np.sign(x0[-1] - x0[0])
        if net_direction > 0:
            x0 = -x0
            y0 = -y0

        sims.append({
            "episode": ep,
            "traj_x": resample_to_fixed_length(x0),
            "traj_y": resample_to_fixed_length(y0),
        })
        print(f"episode {ep}: {len(com_x_traj)} steps")
    return sims


def band(fig, x, mean, std, color, name):
    fig.add_trace(go.Scatter(x=x, y=mean, mode="lines", name=name,
                             line=dict(color=color)))
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([mean + std, (mean - std)[::-1]]),
        fill="toself", fillcolor=color, opacity=0.15,
        line=dict(width=0), showlegend=False))


def plot_comy_timeseries(sims, trials, cfg_key):
    x = np.linspace(0, 1, RESAMPLE_LEN)
    sim_y = np.array([s["traj_y"] for s in sims])
    hum_y = np.array([t["traj_y"] for t in trials])

    fig = go.Figure()
    band(fig, x, sim_y.mean(axis=0), sim_y.std(axis=0), "royalblue", "sim mean")
    band(fig, x, hum_y.mean(axis=0), hum_y.std(axis=0), "firebrick", "human mean")
    fig.update_layout(
        title=f"Sim vs. human mean CoM-y (zero-referenced) -- {cfg_key}<br>"
              f"<sup>n_sim={len(sims)}, n_human={len(trials)}, "
              f"fixed target={FIXED_TARGET}</sup>",
        xaxis_title="normalized time",
        yaxis_title="CoM-y displacement from trial start (m)")
    fig.write_html(f"comy_comparison_{cfg_key}.html")
    print(f"Saved comy_comparison_{cfg_key}.html")


def plot_xy_path(sims, trials, cfg_key):
    """The supervisor's stated alternative: the x/y CoM plot. Shows the
    2D path traced by the CoM in the horizontal plane."""
    sim_x = np.array([s["traj_x"] for s in sims])
    sim_y = np.array([s["traj_y"] for s in sims])
    hum_x = np.array([t["traj_x"] for t in trials])
    hum_y = np.array([t["traj_y"] for t in trials])

    fig = go.Figure()
    # individual trials, faint
    for t in trials:
        fig.add_trace(go.Scatter(x=t["traj_x"], y=t["traj_y"], mode="lines",
                                 line=dict(color="firebrick", width=1),
                                 opacity=0.08, showlegend=False))
    for s in sims:
        fig.add_trace(go.Scatter(x=s["traj_x"], y=s["traj_y"], mode="lines",
                                 line=dict(color="royalblue", width=1),
                                 opacity=0.15, showlegend=False))
    # means, bold
    fig.add_trace(go.Scatter(x=hum_x.mean(axis=0), y=hum_y.mean(axis=0),
                             mode="lines", name="human mean path",
                             line=dict(color="firebrick", width=4)))
    fig.add_trace(go.Scatter(x=sim_x.mean(axis=0), y=sim_y.mean(axis=0),
                             mode="lines", name="sim mean path",
                             line=dict(color="royalblue", width=4)))
    fig.update_layout(
        title=f"CoM path in horizontal plane, sim vs. human -- {cfg_key}<br>"
              f"<sup>zero-referenced to trial start; faint lines are "
              f"individual trials/episodes</sup>",
        xaxis_title="CoM-x displacement (m)",
        yaxis_title="CoM-y displacement (m)")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.write_html(f"comxy_path_{cfg_key}.html")
    print(f"Saved comxy_path_{cfg_key}.html")


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


if __name__ == "__main__":
    cfg_key = sys.argv[1] if len(sys.argv) > 1 else list(CONFIGS.keys())[0]

    trials = load_human_trials()
    sims = collect_sim_trajectories(cfg_key)
    print(f"\nHuman trials: {len(trials)}   Sim episodes: {len(sims)}")

    # numeric summary on the Y axis, mirroring the x-axis pipeline's format
    corrs_y, rmses_y = [], []
    for t in trials:
        cs = [corr(t["traj_y"], s["traj_y"]) for s in sims]
        rs = [rmse(t["traj_y"], s["traj_y"]) for s in sims]
        corrs_y.append(np.mean(cs))
        rmses_y.append(np.mean(rs))

    print("\n=== CoM-Y SUMMARY (zero-referenced, fixed target) ===")
    print(f"Mean correlation (all sims): {np.mean(corrs_y):.4f}")
    print(f"Mean RMSE (all sims):        {np.mean(rmses_y):.4f} m")
    print("\nNOTE: sim com_y is actively penalised toward zero by "
          "com_y_penalty and has no target or perturbation on this axis. "
          "A low correlation here may reflect that design choice rather "
          "than a policy deficiency -- see the x/y path plot instead.")

    # magnitude sanity check -- is the sim's y motion even comparable in scale?
    sim_y_range = np.mean([np.ptp(s["traj_y"]) for s in sims])
    hum_y_range = np.mean([np.ptp(t["traj_y"]) for t in trials])
    print(f"\nMean peak-to-peak CoM-y excursion:")
    print(f"  sim:   {sim_y_range:.4f} m")
    print(f"  human: {hum_y_range:.4f} m")
    if sim_y_range > 0 and hum_y_range / sim_y_range > 5:
        print("  -> human y excursion is >5x the sim's. The sim is holding "
              "y near zero as instructed by com_y_penalty; the axes are not "
              "doing comparable things.")

    plot_comy_timeseries(sims, trials, cfg_key)
    plot_xy_path(sims, trials, cfg_key)
