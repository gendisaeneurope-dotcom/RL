"""Train one configuration. Parallel envs, seed-controlled, screening-length by default.

  python train.py --mode preliminary --seed 0                     # baseline
  python train.py --mode target --safety none    --weight 0   --seed 0
  python train.py --mode target --safety xcom    --weight 1.0 --seed 0
  python train.py --mode target --safety capture --weight 1.0 --seed 0

Preliminary uses a disturbance curriculum by default (matching the original
baseline script's 4 stages); pass --no-curriculum to train with a single
fixed --disturb-prob instead.
"""
import argparse, os, time, json, numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.logger import configure
from postural_env import PosturalEnv

CURRICULUM = [
    {"disturb_prob": 0.0,  "timesteps": 100_000},
    {"disturb_prob": 0.05, "timesteps": 150_000},
    {"disturb_prob": 0.15, "timesteps": 150_000},
    {"disturb_prob": 0.3,  "timesteps": 100_000},
]


def make_env(mode, safety, weight, seed, rank, log_dir, disturb_prob=0.0, use_shaping=False):
    def _f():
        e = PosturalEnv(mode=mode, safety=safety, safety_weight=weight,
                        disturb_prob=disturb_prob, use_shaping=use_shaping)
        e = TimeLimit(e, max_episode_steps=1000)
        # filename per rank so Monitor's csv writer doesn't collide across
        # the parallel envs; load_results() in plot_results.py reads all of
        # them together automatically.
        e = Monitor(e, os.path.join(log_dir, f"monitor_{rank}"))
        e.reset(seed=seed + rank)
        return e
    return _f


def build_model(env, seed):
    model = PPO("MlpPolicy", env, n_steps=512, batch_size=512, ent_coef=0.01,
                learning_rate=3e-4, gamma=0.99, seed=seed, verbose=1)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="target", choices=["target", "preliminary"])
    p.add_argument("--safety", choices=["none", "xcom", "capture", "joint"], default="none")
    p.add_argument("--weight", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000,
                    help="Ignored in preliminary mode unless --no-curriculum is set.")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--no-curriculum", action="store_true",
                    help="Preliminary mode: train at a single fixed --disturb-prob "
                         "instead of the 4-stage curriculum.")
    p.add_argument("--disturb-prob", type=float, default=0.0,
                    help="Used only with --no-curriculum in preliminary mode.")
    p.add_argument("--use-shaping", action="store_true",
                    help="Re-enable the shaping_bonus term (not in the paper's "
                         "EC_omega, but 100k-step sanity check showed reward/step "
                         "improving -8.48->-4.52 WITH it vs flat/worse WITHOUT it. "
                         "Off by default to match the paper's clean 2-term convex "
                         "combination; pass this flag to override that choice.")
    a = p.parse_args()

    if a.mode == "preliminary":
        tag = f"preliminary_s{a.seed}"
    else:
        tag = f"{a.safety}_w{a.weight:g}_s{a.seed}"
    log_dir = f"./runs/{tag}/"
    os.makedirs(log_dir, exist_ok=True)

    if a.mode == "preliminary" and not a.no_curriculum:
        # Re-create the SubprocVecEnv at each stage with the new
        # disturb_prob, matching the original script's curriculum loop.
        model = None
        total_done = 0
        for stage in CURRICULUM:
            env = SubprocVecEnv([make_env(a.mode, a.safety, a.weight, a.seed * 1000, i,
                                          log_dir, disturb_prob=stage["disturb_prob"],
                                          use_shaping=a.use_shaping)
                                 for i in range(a.n_envs)])
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            if model is None:
                model = build_model(env, a.seed)
                model.set_logger(configure(log_dir, ["stdout", "csv"]))
            else:
                model.set_env(env)
            t0 = time.time()
            model.learn(total_timesteps=stage["timesteps"], reset_num_timesteps=False)
            total_done += stage["timesteps"]
            print(f"[{tag}] stage disturb_prob={stage['disturb_prob']} done "
                  f"({time.time()-t0:.0f}s, {total_done} total steps)")
            model.save(f"{log_dir}/model")
            env.save(f"{log_dir}/vecnormalize.pkl")
            env.close()
        with open(f"{log_dir}/config.json", "w") as f:
            json.dump({"mode": a.mode, "safety": a.safety, "weight": a.weight,
                       "use_shaping": a.use_shaping}, f)
        return

    disturb = a.disturb_prob if a.mode == "preliminary" else 0.0
    env = SubprocVecEnv([make_env(a.mode, a.safety, a.weight, a.seed * 1000, i, log_dir,
                                  disturb_prob=disturb, use_shaping=a.use_shaping)
                         for i in range(a.n_envs)])
    # norm_obs=True: the previous script had this OFF, leaving target_y as the
    # smallest-magnitude input in the observation vector.
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = build_model(env, a.seed)
    model.set_logger(configure(log_dir, ["stdout", "csv"]))

    t0 = time.time()
    model.learn(total_timesteps=a.steps)
    dt = time.time() - t0
    print(f"\n[{tag}] {a.steps} steps in {dt/60:.1f} min "
          f"({a.steps/dt:.0f} fps)")

    model.save(f"{log_dir}/model")
    env.save(f"{log_dir}/vecnormalize.pkl")
    env.close()
    with open(f"{log_dir}/config.json", "w") as f:
        json.dump({"mode": a.mode, "safety": a.safety, "weight": a.weight,
                   "use_shaping": a.use_shaping}, f)


if __name__ == "__main__":
    main()