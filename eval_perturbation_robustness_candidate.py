"""
Escalating perturbation robustness check. Ported from the original
eval_perturbation_robustness.py -- same 5 conditions, same two-part report
(reward/length summary, then explicit target-vs-final-com_y per episode).

Adapted for the standalone candidate1_target.py / candidate2_xcom.py, which
don't have postural_env.py's run_dir/config.json/model structure -- uses the
same CONFIGS dict pattern as plot_candidate.py / why_failed_candidate.py.

Model is always loaded as trained WITHOUT perturbation; disturbance is
introduced here, at test time only.

    python eval_perturbation_robustness_candidate.py candidate1
    python eval_perturbation_robustness_candidate.py candidate2_xcom
"""
import sys
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

CONFIGS = {
    "candidate1_A": dict(module="candidate1_A", cls="Candidate1Env",
                        model="ppo_candidate1_A", vecnorm="vecnormalize_candidate1_A.pkl",
                        log_dir="./training_logs_candidate1_A/"),
    "candidate2_xcom": dict(module="candidate2_xcom", cls="Candidate2Env",
                             model="ppo_candidate2_xcom", vecnorm="vecnormalize_candidate2_xcom.pkl",
                             log_dir="./training_logs_candidate2_xcom/"),
    "candidate1_B": dict(module="candidate1_B", cls="Candidate1Env",
                          model="ppo_candidate1_B", vecnorm="vecnormalize_candidate1_B.pkl",
                          log_dir="./training_logs_candidate1_B/"),
    "candidate1_C": dict(module="candidate1_C", cls="Candidate1Env",
                          model="ppo_candidate1_C", vecnorm="vecnormalize_candidate1_C.pkl",
                          log_dir="./training_logs_candidate1_C/"),
    "candidate1_D": dict(module="candidate1_D", cls="Candidate1Env",
                          model="ppo_candidate1_D", vecnorm="vecnormalize_candidate1_D.pkl",
                          log_dir="./training_logs_candidate1_D/"),
    "candidate1_E": dict(module="candidate1_E", cls="Candidate1Env",
                              model="ppo_candidate1_E", vecnorm="vecnormalize_candidate1_E.pkl",
                              log_dir="./training_logs_candidate1_E/"),
    "candidate1_F": dict(module="candidate1_F", cls="Candidate1Env",
                                  model="ppo_candidate1_F", vecnorm="vecnormalize_candidate1_F.pkl",
                                  log_dir="./training_logs_candidate1_F/"),
}

# Same escalating conditions as the original script.
CONDITIONS = [
    {"disturb_prob": 0.0, "force_range": (-20, 20)},     # baseline, no disturbance
    {"disturb_prob": 0.05, "force_range": (-20, 20)},    # light, occasional
    {"disturb_prob": 0.3, "force_range": (-20, 20)},     # moderate, same force
    {"disturb_prob": 1.0, "force_range": (-20, 20)},     # constant, same force
    {"disturb_prob": 1.0, "force_range": (-100, 100)},   # constant, much stronger
]


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


def run_one(key):
    cfg = CONFIGS[key]
    module = __import__(cfg["module"])
    EnvClass = getattr(module, cfg["cls"])
    model = PPO.load(cfg["model"])

    print(f"\n{'#'*60}\n# {key}\n{'#'*60}")

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
            print(f"  target_y={i['target_y']:.4f}, final com_y={i['com_y']:.4f}, "
                  f"error={err:.4f}, failed={i['failed']}")
            venv2.close()
        print(f"Mean |error| across 10 episodes: {np.mean(np.abs(errors)):.4f}")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    keys = list(CONFIGS.keys()) if arg == "all" else [arg]
    for key in keys:
        try:
            run_one(key)
        except FileNotFoundError as e:
            print(f"\nSkipping {key}: missing model/vecnorm file ({e})")
        except Exception as e:
            print(f"\nSkipping {key}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()