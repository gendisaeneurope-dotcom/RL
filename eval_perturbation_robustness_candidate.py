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
    "candidate1": dict(module="candidate1_target", cls="Candidate1Env",
                        model="ppo_candidate1", vecnorm="vecnormalize_candidate1.pkl"),
    "candidate2_xcom": dict(module="candidate2_xcom", cls="Candidate2Env",
                             model="ppo_candidate2_xcom", vecnorm="vecnormalize_candidate2_xcom.pkl"),
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
        if seed is not None:
            e.reset(seed=seed)
        return e
    venv = DummyVecEnv([_f])
    venv = VecNormalize.load(cfg["vecnorm"], venv)
    venv.training = False
    venv.norm_reward = False
    return venv


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "candidate1"
    cfg = CONFIGS[key]
    module = __import__(cfg["module"])
    EnvClass = getattr(module, cfg["cls"])

    model = PPO.load(cfg["model"])
    # Both candidate envs are target-reaching (unlike postural_env.py's
    # "preliminary" mode), so the target-tracking check always applies here.

    for cond in CONDITIONS:
        print(f"\n=== disturb_prob={cond['disturb_prob']}, "
              f"force_range={cond['force_range']} ===")

        # Reward/length summary
        venv = make_venv(EnvClass, cfg, cond)
        rewards, lengths = evaluate_policy(model, venv, n_eval_episodes=20,
                                           return_episode_rewards=True)
        venv.close()
        print(f"Mean reward: {np.mean(rewards):.4f} +/- {np.std(rewards):.4f}  |  "
              f"episodes at full length: {sum(1 for l in lengths if l == 1000)}/20")

        # Does it still reach the target despite the disturbance? Reward/
        # length alone can't answer this -- need final com_y vs target_y.
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


if __name__ == "__main__":
    main()