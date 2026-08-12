"""
compare_heightcheck_vs_original.py
======================================
DIRECT TEST OF H4: does the missing upright/height success condition
actually matter in practice, or do joint-limit/capture-point terminations
already preclude the relevant failure state?

Runs BOTH models (existing, unmodified Candidate 2, and the new
height-check variant) through IDENTICAL seeds and the SAME fixed target,
and logs per-episode: success (True/False), final tracking error, and
whether is_upright was ever False during the episode (height-check model
only -- the original model doesn't track this).

Usage:
    python compare_heightcheck_vs_original.py
"""
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import TimeLimit

N_EPISODES = 40
FIXED_TARGET = 0.08  # same data-derived target used elsewhere

MODELS = {
    "Original (no height check)": dict(
        module="candidate2_ap_comy1_staypenalty",
        env_class="Candidate2Env",
        model="ppo_candidate2_ap_comy1_staypenalty_6",
        vecnorm="vecnormalize_candidate2_ap_comy1_staypenalty_6.pkl",
    ),
    "Height-check variant": dict(
        module="train_height_check",
        env_class="Candidate2EnvHeightCheck",
        model="ppo_candidate2_heightcheck_run2",
        vecnorm="vecnormalize_candidate2_heightcheck_run2.pkl",
    ),
}


def evaluate_model(module_name, env_class_name, model_path, vecnorm_path,
                    n_episodes=N_EPISODES):
    module = __import__(module_name)
    EnvClass = getattr(module, env_class_name)
    model = PPO.load(model_path)

    rows = []
    for ep in range(n_episodes):
        # SAME seed used across both models for a matched comparison.
        env_instance = EnvClass(fixed_target=FIXED_TARGET)
        venv = DummyVecEnv([lambda: TimeLimit(env_instance, max_episode_steps=1000)])
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
        venv.seed(ep)  # matched across both models -- same seed value per episode index
        obs = venv.reset()

        done = False
        final_info = None
        ever_not_upright = False
        step_count = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done_v, info = venv.step(action)
            done = bool(done_v[0])
            final_info = info[0]
            step_count += 1
            if "is_upright" in final_info and not final_info["is_upright"]:
                ever_not_upright = True
        venv.close()

        final_tracking_error = abs(final_info["com_x"] - final_info["target_x"])
        rows.append({
            "episode": ep,
            "success": final_info.get("success", None),
            "failed": final_info.get("failed", None),
            "final_tracking_error": final_tracking_error,
            "episode_length": step_count,
            "ever_not_upright_but_otherwise_ok": ever_not_upright,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_results = {}
    for name, cfg in MODELS.items():
        print(f"=== Evaluating: {name} ===")
        try:
            df = evaluate_model(cfg["module"], cfg["env_class"], cfg["model"], cfg["vecnorm"])
            all_results[name] = df

            success_rate = df["success"].mean() if df["success"].notna().any() else float("nan")
            fail_rate = df["failed"].mean() if df["failed"].notna().any() else float("nan")
            mean_final_error = df["final_tracking_error"].mean()
            n_not_upright = df["ever_not_upright_but_otherwise_ok"].sum()

            print(f"  Success rate: {success_rate:.3f}")
            print(f"  Fail rate: {fail_rate:.3f}")
            print(f"  Mean final tracking error: {mean_final_error:.5f} m")
            if n_not_upright > 0:
                print(f"  Episodes where is_upright was False at some point: {n_not_upright}/{len(df)}")
            print()

            df.to_csv(f"heightcheck_eval_{name.replace(' ', '_').replace('(', '').replace(')', '')}.csv",
                       index=False)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}\n")

    if len(all_results) == 2:
        print("=== DIRECT COMPARISON ===")
        names = list(all_results.keys())
        df1, df2 = all_results[names[0]], all_results[names[1]]

        succ1 = df1["success"].mean() if df1["success"].notna().any() else float("nan")
        succ2 = df2["success"].mean() if df2["success"].notna().any() else float("nan")
        err1 = df1["final_tracking_error"].mean()
        err2 = df2["final_tracking_error"].mean()

        print(f"Success rate: {names[0]}={succ1:.3f} vs. {names[1]}={succ2:.3f} "
              f"(diff={abs(succ1-succ2):.3f})")
        print(f"Mean final tracking error: {names[0]}={err1:.5f} vs. {names[1]}={err2:.5f} "
              f"(diff={abs(err1-err2):.5f} m)")
        print()
        print("INTERPRETATION:")
        print("  If both numbers are close (small diff), H4 is SUPPORTED -- the missing")
        print("  upright check did not matter in practice, because joint-limit/capture-")
        print("  point terminations already precluded the relevant failure state.")
        print("  If success rate or tracking error differs substantially, H4 is REFUTED --")
        print("  you've found a genuine, previously-hidden gap in the original success")
        print("  condition worth reporting as a real finding.")
        print()
        print("CAUTION (per last night's A_SCALE result): a single run of each model may")
        print("not be reliable on its own. If you have time, retrain the height-check")
        print("variant 1-2 more times with different seeds and re-run this comparison")
        print("before treating either conclusion as final.")
