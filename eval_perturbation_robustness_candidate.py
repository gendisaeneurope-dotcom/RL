"""
Escalating perturbation robustness check. Ported from the original
eval_perturbation_robustness.py -- same 5 conditions, same two-part report
(reward/length summary, then explicit target-vs-final-com_y per episode).

Adapted for the standalone candidate1_target.py / candidate2_xcom.py, which
don't have postural_env.py's run_dir/config.json/model structure -- uses the
same CONFIGS dict pattern as plot_candidate.py / why_failed_candidate.py.

Model is always loaded as trained WITHOUT perturbation; disturbance is
introduced here, at test time only.

    python eval_perturbation_robustness_candidate.py candidate1_F
    python eval_perturbation_robustness_candidate.py candidate2_xcom
    python eval_perturbation_robustness_candidate.py candidate3_capturepoint
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
    "candidate1_F": dict(module="candidate1_F", cls="Candidate1Env",
                                  model="ppo_candidate1_F", vecnorm="vecnormalize_candidate1_F.pkl",
                                  log_dir="./training_logs_candidate1_F/"),
    "candidate2_xcom": dict(module="candidate2_xcom", cls="Candidate2Env",
                             model="ppo_candidate2_xcom", vecnorm="vecnormalize_candidate2_xcom.pkl",
                             log_dir="./training_logs_candidate2_xcom/"),
    "candidate3_capturepoint": dict(module="candidate3_capturepoint", cls="Candidate3Env",
                                     model="ppo_candidate3_capturepoint", vecnorm="vecnormalize_candidate3_capturepoint.pkl",
                                     log_dir="./training_logs_candidate3_capturepoint/"),
}

# Same escalating conditions as the original script.
CONDITIONS = [
    {"disturb_prob": 0.0, "force_range": (-20, 20)},     # baseline, no disturbance
    {"disturb_prob": 0.05, "force_range": (-20, 20)},    # light, occasional
    {"disturb_prob": 0.3, "force_range": (-20, 20)},     # moderate, same force
    {"disturb_prob": 1.0, "force_range": (-20, 20)},     # constant, same force
    {"disturb_prob": 1.0, "force_range": (-100, 100)},   # constant, much stronger
]

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


# Label for x-axis: combine disturb_prob and force_range into one readable tick
def cond_label(cond):
    return f"p={cond['disturb_prob']}\nF={cond['force_range'][1]}N"

def run_one(key, results_store):
    cfg = CONFIGS[key]
    module = __import__(cfg["module"])
    EnvClass = getattr(module, cfg["cls"])
    model = PPO.load(cfg["model"])

    print(f"\n{'#'*60}\n# {key}\n{'#'*60}")

    mean_rewards, mean_abs_errors, labels = [], [], []

    for cond in CONDITIONS:
        print(f"\n=== disturb_prob={cond['disturb_prob']}, "
              f"force_range={cond['force_range']} ===")

        venv = make_venv(EnvClass, cfg, cond)
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
            err = i["com_y"] - i["target_y"]
            errors.append(err)
            venv2.close()

        mean_rewards.append(float(np.mean(rewards)))
        mean_abs_errors.append(float(np.mean(np.abs(errors))))
        labels.append(cond_label(cond))

    results_store[key] = dict(labels=labels, mean_rewards=mean_rewards,
                               mean_abs_errors=mean_abs_errors)


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
                             xaxis_title="Disturbance condition", yaxis_title="Mean |com_y - target_y| (m)")

        for fig, name in [(fig_r, "perturbation_reward"), (fig_e, "perturbation_error")]:
            path = os.path.join(candidate_dir, name)
            try:
                fig.write_image(path + ".png")
            except Exception as e:
                fig.write_html(path + ".html")
                print(f"PNG export failed ({e}), wrote HTML instead.")

        # add this candidate's line to the combined overlay figures too
        fig_r_all.add_trace(go.Scatter(x=res["labels"], y=res["mean_rewards"],
                                        mode="lines+markers", name=key))
        fig_e_all.add_trace(go.Scatter(x=res["labels"], y=res["mean_abs_errors"],
                                        mode="lines+markers", name=key))

    fig_r_all.update_layout(title="Mean reward vs. disturbance severity (all candidates)",
                             xaxis_title="Disturbance condition", yaxis_title="Mean episode reward")
    fig_e_all.update_layout(title="Mean |tracking error| vs. disturbance severity (all candidates)",
                             xaxis_title="Disturbance condition", yaxis_title="Mean |com_y - target_y| (m)")

    for fig, name in [(fig_r_all, "all_candidates_reward"), (fig_e_all, "all_candidates_error")]:
        path = os.path.join(PLOT_DIR, name)
        try:
            fig.write_image(path + ".png")
        except Exception as e:
            fig.write_html(path + ".html")
            print(f"PNG export failed ({e}), wrote HTML instead.")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    keys = list(CONFIGS.keys()) if arg == "all" else [arg]
    results_store = {}
    for key in keys:
        try:
            run_one(key, results_store)
        except FileNotFoundError as e:
            print(f"\nSkipping {key}: missing model/vecnorm file ({e})")
        except Exception as e:
            print(f"\nSkipping {key}: {type(e).__name__}: {e}")
    plot_all(results_store)
    print(f"\nPerturbation plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()