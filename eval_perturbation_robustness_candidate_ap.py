"""
Escalating perturbation robustness check. Ported from the original
eval_perturbation_robustness.py -- same 5 conditions, same two-part report
(reward/length summary, then explicit target-vs-final-com_y per episode).

Adapted for the standalone candidate1_target.py / candidate2_xcom.py, which
don't have postural_env.py's run_dir/config.json/model structure -- uses the
same CONFIGS dict pattern as plot_candidate.py / why_failed_candidate.py.

    python eval_perturbation_robustness_candidate_ap.py candidate1_F_ap
    python eval_perturbation_robustness_candidate_ap.py candidate2_ap
    python eval_perturbation_robustness_candidate_ap.py candidate3_ap
"""
import sys
import numpy as np
import os
import plotly.graph_objects as go
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


CONFIGS = {
        "candidate1": dict(module="candidate1_ap_comy", cls="Candidate1Env",
                                        model="ppo_candidate1_ap_comy1_staypenalty6",
                                        vecnorm="vecnormalize_candidate1_ap_comy1_staypenalty6.pkl"),
        "candidate2": dict(module="candidate2_ap_comy1_staypenalty_jointfix", cls="Candidate2Env",
                                        model="ppo_candidate2_ap_comy1_staypenalty_jointfix",
                                        vecnorm="vecnormalize_candidate2_ap_comy1_staypenalty_jointfix.pkl"),
        "candidate3": dict(module="candidate3_ap_comy1_staypenalty", cls="Candidate3Env",
                                        model="ppo_candidate3_ap_comy1_staypenalty6",
                                        vecnorm="vecnormalize_candidate3_ap_comy1_staypenalty6.pkl"),
    }

# Same escalating conditions as the original script -- used for the reward/error
# severity sweep only. NOT used for the single-condition trajectory plot below.
CONDITIONS = [
    {"disturb_prob": 0.0, "force_range": (-20, 20)},     # baseline, no disturbance
    {"disturb_prob": 0.05, "force_range": (-20, 20)},    # light, occasional
    {"disturb_prob": 0.3, "force_range": (-20, 20)},     # moderate, same force
    {"disturb_prob": 1.0, "force_range": (-20, 20)},     # constant, same force
    {"disturb_prob": 1.0, "force_range": (-100, 100)},   # constant, much stronger
]

# Single condition used for the trajectory (time/xy) plots, kept in sync with
# plot_candidate_ap.py's --perturb setting so the two scripts' trajectory
# plots are directly comparable.
TRAJ_DISTURB_PROB = 0.0
TRAJ_FORCE_RANGE = (-30, 30)

PLOT_DIR = "./perturbation_plots/"
os.makedirs(PLOT_DIR, exist_ok=True)


def make_venv(EnvClass, cfg, cond, seed=None):
    def _f():
        e = EnvClass(**cond)
        e = TimeLimit(e, max_episode_steps=1000)
        return e
    venv = DummyVecEnv([_f])
    venv = VecNormalize.load(cfg["vecnorm"], venv)
    venv.training = False
    venv.norm_reward = False
    if seed is not None:
        venv.seed(seed)
    return venv


def cond_label(cond):
    return f"p={cond['disturb_prob']}\nF={cond['force_range'][1]}N"


def run_one(key, results_store, seed=None):
    cfg = CONFIGS[key]
    module = __import__(cfg["module"])
    EnvClass = getattr(module, cfg["cls"])
    model = PPO.load(cfg["model"])

    print(f"\n{'#'*60}\n# {key}\n{'#'*60}")

    mean_rewards, mean_abs_errors, labels = [], [], []

    for cond in CONDITIONS:
        print(f"\n=== disturb_prob={cond['disturb_prob']}, "
              f"force_range={cond['force_range']} ===")

        venv = make_venv(EnvClass, cfg, cond, seed=seed)
        rewards, lengths = evaluate_policy(model, venv, n_eval_episodes=20,
                                           return_episode_rewards=True)
        venv.close()
        print(f"Mean reward: {np.mean(rewards):.4f} +/- {np.std(rewards):.4f}  |  "
              f"episodes at full length: {sum(1 for l in lengths if l == 1000)}/20")

        errors = []
        for ep in range(10):
            venv2 = make_venv(EnvClass, cfg, cond, seed=ep)
            obs = venv2.reset()
            info = [{}]
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done_v, info = venv2.step(action)
                done = bool(done_v[0])
            i = info[0]
            err = i["com_x"] - i["target_x"]
            errors.append(err)
            venv2.close()

        mean_rewards.append(float(np.mean(rewards)))
        mean_abs_errors.append(float(np.mean(np.abs(errors))))
        labels.append(cond_label(cond))

    results_store[key] = dict(labels=labels, mean_rewards=mean_rewards,
                               mean_abs_errors=mean_abs_errors)

    candidate_dir = os.path.join(PLOT_DIR, key)
    plot_perturbation_trajectories(EnvClass, model, cfg["vecnorm"], out_dir=candidate_dir,
                                     disturb_prob=TRAJ_DISTURB_PROB, force_range=TRAJ_FORCE_RANGE)


def plot_all(results_store):
    """Per-candidate plots (own subfolder) AND one combined overlay per metric."""
    fig_r_all = go.Figure()
    fig_e_all = go.Figure()

    for key, res in results_store.items():
        candidate_dir = os.path.join(PLOT_DIR, key)
        os.makedirs(candidate_dir, exist_ok=True)

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=res["labels"], y=res["mean_rewards"],
                                    mode="lines+markers", name=key))
        fig_r.update_layout(title=f"Mean reward vs. disturbance severity ({key})",
                             xaxis_title="Disturbance condition", yaxis_title="Mean episode reward")

        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(x=res["labels"], y=res["mean_abs_errors"],
                                    mode="lines+markers", name=key))
        fig_e.update_layout(title=f"Mean |tracking error| vs. disturbance severity ({key})",
                             xaxis_title="Disturbance condition", yaxis_title="Mean |com_x - target_x| (m)")

        for fig, name in [(fig_r, "perturbation_reward"), (fig_e, "perturbation_error")]:
            path = os.path.join(candidate_dir, name)
            try:
                fig.write_image(path + ".png")
            except Exception as e:
                fig.write_html(path + ".html")
                print(f"PNG export failed ({e}), wrote HTML instead.")

        fig_r_all.add_trace(go.Scatter(x=res["labels"], y=res["mean_rewards"],
                                        mode="lines+markers", name=key))
        fig_e_all.add_trace(go.Scatter(x=res["labels"], y=res["mean_abs_errors"],
                                        mode="lines+markers", name=key))

    fig_r_all.update_layout(title="Mean reward vs. disturbance severity (all candidates)",
                             xaxis_title="Disturbance condition", yaxis_title="Mean episode reward")
    fig_e_all.update_layout(title="Mean |tracking error| vs. disturbance severity (all candidates)",
                             xaxis_title="Disturbance condition", yaxis_title="Mean |com_x - target_x| (m)")

    for fig, name in [(fig_r_all, "all_candidates_reward"), (fig_e_all, "all_candidates_error")]:
        path = os.path.join(PLOT_DIR, name)
        try:
            fig.write_image(path + ".png")
        except Exception as e:
            fig.write_html(path + ".html")
            print(f"PNG export failed ({e}), wrote HTML instead.")


def plot_perturbation_trajectories(env_cls, model, vecnorm_path, out_dir,
                                     disturb_prob=TRAJ_DISTURB_PROB, force_range=TRAJ_FORCE_RANGE, episodes=10):
    venv = DummyVecEnv([lambda: TimeLimit(
        env_cls(disturb_prob=disturb_prob, force_range=force_range),
        max_episode_steps=1000)])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False

    fig_time = go.Figure()
    fig_xy = go.Figure()

    for ep in range(episodes):
        obs = venv.reset()
        done, step = False, 0
        xs, ys, steps = [], [], []
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = venv.step(act)
            done = done[0]
            i = info[0]
            xs.append(i["com_x"])
            ys.append(i.get("com_y", 0.0))
            steps.append(step)
            step += 1
        fig_time.add_trace(go.Scatter(x=steps, y=xs, mode="lines", opacity=0.6,
                                       name=f"ep {ep+1}"))
        fig_xy.add_trace(go.Scatter(x=xs, y=ys, mode="lines", opacity=0.6,
                                     name=f"ep {ep+1}"))
    venv.close()

    fig_time.update_layout(title=f"CoM-x over time under perturbation (prob={disturb_prob}, force={force_range})",
                            xaxis_title="step", yaxis_title="CoM-x (m)")
    fig_xy.update_layout(title=f"CoM x/y trajectory under perturbation (prob={disturb_prob}, force={force_range})",
                          xaxis_title="CoM-x (m)", yaxis_title="CoM-y (m)")

    os.makedirs(out_dir, exist_ok=True)
    fig_time.write_image(os.path.join(out_dir, "perturbation_com_over_time.png"))
    fig_xy.write_image(os.path.join(out_dir, "perturbation_com_xy.png"))
    print(f"Saved perturbation time/xy plots to {out_dir}/")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
    keys = list(CONFIGS.keys()) if arg == "all" else [arg]
    results_store = {}
    for key in keys:
        try:
            run_one(key, results_store, seed=seed)
        except FileNotFoundError as e:
            print(f"\nSkipping {key}: missing model/vecnorm file ({e})")
        except Exception as e:
            print(f"\nSkipping {key}: {type(e).__name__}: {e}")
    plot_all(results_store)
    print(f"\nPerturbation plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
