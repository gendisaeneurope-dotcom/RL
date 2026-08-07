"""
Adapter to reuse why_failed.py's diagnostic logic against the standalone
candidate1_target.py / candidate2_xcom.py outputs, which don't share
postural_env.py's module-level JOINT_RANGE / FAIL_MARGIN constants.

Usage:
    python why_failed_candidate_ap.py candidate1_ap
    python why_failed_candidate_ap.py candidate2_ap
    python why_failed_candidate_ap.py candidate3_ap
"""
import sys
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

JOINT_NAMES = ["ankle_eversion", "ankle_flexion", "hip_abduction", "hip_flexion"]

CONFIGS = {
    "candidate1_ap": dict(module="candidate1_ap", cls="Candidate1Env",
                             model="ppo_candidate1_F_ap", vecnorm="vecnormalize_candidate1_F_ap.pkl",
                             log_dir="./training_logs_candidate1_F_ap/"),
    "candidate1_ap_comy1": dict(module="candidate1_ap", cls="Candidate1Env",
                                     model="ppo_candidate1_ap_comy1", vecnorm="vecnormalize_candidate1_ap_comy1.pkl",
                                     log_dir="./training_logs_candidate1_ap_comy1/"),                          
    "candidate2_ap": dict(module="candidate2_ap", cls="Candidate2Env",
                           model="ppo_candidate2_ap", vecnorm="vecnormalize_candidate2_ap.pkl",
                           log_dir="./training_logs_candidate2_ap/"),
    "candidate2_ap_comy1": dict(module="candidate2_ap", cls="Candidate2Env",
                               model="ppo_candidate2_ap_comy1", vecnorm="vecnormalize_candidate2_ap_comy1.pkl",
                               log_dir="./training_logs_candidate2_ap_comy1/"),
    "candidate3_ap": dict(module="candidate3_ap", cls="Candidate3Env",
                           model="ppo_candidate3_ap", vecnorm="vecnormalize_candidate3_ap.pkl",
                           log_dir="./training_logs_candidate3_ap/"),
    "candidate3_ap_comy1": dict(module="candidate3_ap", cls="Candidate3Env",
                                model="ppo_candidate3_ap_comy1", vecnorm="vecnormalize_candidate3_ap_comy1.pkl",
                                log_dir="./training_logs_candidate3_ap_comy1/"),
}


def get_joint_limits(probe_env):
    """
    Pull JOINT_LOW/JOINT_HIGH off the env instance or its module, since
    candidate1_target.py / candidate2_xcom.py define these as module-level
    constants (JOINT_LOW, JOINT_HIGH), not as attributes on the env class
    itself. Falls back through a few likely names before giving up.
    """
    module = sys.modules[type(probe_env).__module__]
    for low_name, high_name in [("JOINT_LOW", "JOINT_HIGH"),
                                 ("joint_low", "joint_high")]:
        if hasattr(module, low_name) and hasattr(module, high_name):
            fail_margin = getattr(module, "FAIL_MARGIN", 1.0)
            return (np.asarray(getattr(module, low_name)) * fail_margin,
                    np.asarray(getattr(module, high_name)) * fail_margin)
    # last resort: attributes directly on the instance
    if hasattr(probe_env, "joint_low") and hasattr(probe_env, "joint_high"):
        return np.asarray(probe_env.joint_low), np.asarray(probe_env.joint_high)
    raise AttributeError(
        f"Could not find JOINT_LOW/JOINT_HIGH on module {module.__name__} "
        f"or instance {type(probe_env).__name__}. Check the actual constant names."
    )


def get_target_bounds(probe_env):
    module = sys.modules[type(probe_env).__module__]
    if hasattr(module, "TARGET_X_LOW") and hasattr(module, "TARGET_X_HIGH"):
        return float(module.TARGET_X_LOW), float(module.TARGET_X_HIGH)
    if hasattr(probe_env, "target_x_low") and hasattr(probe_env, "target_x_high"):
        return float(probe_env.target_x_low), float(probe_env.target_x_high)
    raise AttributeError("Could not find TARGET_X_LOW/TARGET_X_HIGH on module or instance.")


def rollout(model, EnvClass, cfg, fixed_target, seed):
    def make():
        e = EnvClass(fixed_target=fixed_target)
        e = TimeLimit(e, max_episode_steps=1000)
        return e

    venv = DummyVecEnv([make])
    venv = VecNormalize.load(cfg["vecnorm"], venv)
    venv.training = False
    venv.norm_reward = False

    venv.seed(seed)         
    obs = venv.reset()    

    raw = venv.venv.envs[0].unwrapped
    n = 0
    q = raw.data.qpos[:4].copy()
    info = [{}]
    for _ in range(1000):
        act, _ = model.predict(obs, deterministic=False)
        q = raw.data.qpos[:4].copy()
        obs, _, done, info = venv.step(act)
        n += 1
        if done[0]:
            break
    venv.close()
    return n, q, info[0]


def run(key):
    cfg = CONFIGS[key]
    module = __import__(cfg["module"])
    EnvClass = getattr(module, cfg["cls"])

    model = PPO.load(cfg["model"])
    probe = EnvClass()
    joint_low, joint_high = get_joint_limits(probe)
    target_x_low, target_x_high = get_target_bounds(probe)

    conditions = [("target", t) for t in np.linspace(target_x_low, target_x_high, 11)]
    header = f"{'target':>8}"
    print(f"{'target':>8} {'seed':>8} {'steps':>6} {'outcome':>8} {'culprit joint':>16} "
          f"[q/limit at end: ev, aflex, abd, hflex]")
    print("-" * 100)

    for kind, val in conditions:
        for seed in range(20):
            n, q, info = rollout(model, EnvClass, cfg, fixed_target=val, seed=seed)

            frac = np.where(
                q >= 0,
                q / joint_high,
                q / joint_low,
            )
            worst = int(np.argmax(np.abs(frac)))
            failed = info.get("failed", False)
            culprit = JOINT_NAMES[worst] if failed else "-"
            print(f"{val:8.4f}  seed={seed}  {n:6d} {'FELL' if failed else 'ok':>8} {culprit:>16} "
              f"[{' '.join(f'{f:.2f}' for f in frac)}]")

    print("\n(last four columns: each joint's angle as a fraction of its own limit.")
    print(" 1.00 means that joint is at the limit that ends the episode.)")
    print(" ev/abd move the body SIDEWAYS -- what the reward is about.")
    print(" aflex/hflex move it FRONT-BACK -- nothing in the reward controls these.")


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "candidate1_ap"
    run(key)