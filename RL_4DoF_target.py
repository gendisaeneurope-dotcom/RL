import os

import mujoco
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from gymnasium import spaces
import numpy as np


xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ankle_hip_2x2dof.xml")
_base_env = gym.make("InvertedPendulum-v5", xml_file=xml_path).unwrapped
AnkleHipEnv = type(_base_env)

JOINT_NAMES = ["ankle_eversion", "ankle_flexion", "hip_abduction", "hip_flexion"]
N_JOINTS = 4
JOINT_LOW = np.array([-0.3, -0.5, -0.5, -0.5])
JOINT_HIGH = np.array([0.3, 0.5, 0.5, 0.5])

# PLACEHOLDER, not from real experimental data -- verified computationally that
# +-0.259m is the full joint-limit-achievable CoM-y range for this model, but
# targets that close to the limit are unrealistic/unstable to train against.
# Using a conservative fraction until real trial-target magnitudes are available.
TARGET_RANGE = 0.05  # meters, mediolateral (CoM-y)


class My4DOFTargetEnv(AnkleHipEnv):
    def __init__(self, disturb_prob=0.0, force_range=(-20, 20), target_range=TARGET_RANGE,
                 fixed_target=None, **kwargs):
        super().__init__(xml_file=xml_path, **kwargs)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(N_JOINTS,), dtype=np.float32)
        # Observation now includes the target: [angles(4), angvels(4), target_y(1)] = 9-dim.
        # Without this the policy has no way to know what it's aiming for.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 * N_JOINTS + 1,), dtype=np.float64)
        self._current_step = 0
        self._max_steps = 1000
        self.disturb_prob = disturb_prob
        self.force_range = force_range
        self.target_range = target_range
        self.fixed_target = fixed_target  # set a float to disable randomization for debugging
        self.target_y = 0.0

        # Verified names/ids for the generic ankle_hip_2x2dof.xml model.
        self.trunk_body_id = self.model.body("trunk").id
        self.root_body_id = self.model.body("foot").id

    def reset(self, **kwargs):
        self._current_step = 0
        obs, info = super().reset(**kwargs)
        self.data.qpos[:N_JOINTS] *= 0.1
        self.data.qvel[:N_JOINTS] *= 0.1
        mujoco.mj_forward(self.model, self.data)

        self.target_y = (self.fixed_target if self.fixed_target is not None
                          else np.random.uniform(-self.target_range, self.target_range))

        # NEW: track distance-to-target for potential-based shaping
        com_pos = self.data.subtree_com[self.root_body_id][:2]
        self.prev_distance = abs(com_pos[1] - self.target_y)

        return self._get_obs(), info

    def _get_obs(self):
        angles = self.data.qpos[:N_JOINTS].copy()
        angvels = self.data.qvel[:N_JOINTS].copy()
        return np.concatenate([angles, angvels, [self.target_y]]).astype(np.float64)

    def step(self, action):
        if self.disturb_prob > 0 and np.random.rand() < self.disturb_prob:
            force = np.random.uniform(*self.force_range)
            self.data.xfrc_applied[self.trunk_body_id, 0] = force  # anterior, matches real belt perturbation axis
        else:
            self.data.xfrc_applied[self.trunk_body_id, 0] = 0.0

        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        angles = observation[:N_JOINTS]

        failed = bool(
            not np.isfinite(observation).all()
            or np.any(angles < JOINT_LOW)
            or np.any(angles > JOINT_HIGH)
        )

        self._current_step += 1
        terminated = failed

        if failed:
            print(f"Step {self._current_step}, angles(deg): {np.degrees(angles)}, disturb_prob: {self.disturb_prob}")

        # com_pos[1] = CoM-y = mediolateral (verified computationally: ankle_eversion
        # and hip_abduction move com_pos[1] only; com_pos[0] is AP, driven by the
        # flexion joints). Target-reaching reward uses this axis, matching the real
        # mediolateral CoP-steering task.
        com_pos = self.data.subtree_com[self.root_body_id][:2]
        com_y = com_pos[1]

        # NEW: potential-based shaping -- rewards reducing distance, not just being close
        curr_distance = abs(com_y - self.target_y)
        shaping_term = self.prev_distance - curr_distance
        self.prev_distance = curr_distance

        w1, w2, w3 = 50.0, 0.01, 20.0   # w3 is the new shaping weight, untested placeholder
        target_term = -w1 * (com_y - self.target_y) ** 2
        effort_term = -w2 * np.sum(np.square(action))
        reward = target_term + effort_term + w3 * shaping_term + 0.1

        info = {"com_y": com_y, "target_y": self.target_y}
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info


if __name__ == "__main__":
    log_dir = "./training_logs_4dof_target/"
    os.makedirs(log_dir, exist_ok=True)

    env = TimeLimit(My4DOFTargetEnv(render_mode="human"), max_episode_steps=1000)
    env = Monitor(env, log_dir)

    model = PPO(
        "MlpPolicy", env,
        n_steps=2048,
        batch_size=256,
        ent_coef=0.01,
        learning_rate=3e-4,
        gamma=0.99,
        verbose=1,
    )

    stages = [
        {"disturb_prob": 0.0, "timesteps": 100_000},
        {"disturb_prob": 0.05, "timesteps": 150_000},
        {"disturb_prob": 0.15, "timesteps": 150_000},
        {"disturb_prob": 0.3, "timesteps": 100_000},
    ]

    for stage in stages:
        env.unwrapped.disturb_prob = stage["disturb_prob"]
        model.learn(total_timesteps=stage["timesteps"], reset_num_timesteps=False)
        print(f"Finished stage disturb_prob={stage['disturb_prob']}, env reports: {env.unwrapped.disturb_prob}")
        model.save(f"ppo_ankle_hip_target_disturb_{stage['disturb_prob']}")

    model.save("ppo_ankle_hip_target")
    env.close()

    eval_env = TimeLimit(My4DOFTargetEnv(disturb_prob=0.3), max_episode_steps=1000)
    eval_env = Monitor(eval_env)
    episode_rewards, episode_lengths = evaluate_policy(model, eval_env, n_eval_episodes=20, return_episode_rewards=True)
    for i, (r, l) in enumerate(zip(episode_rewards, episode_lengths)):
        print(f"Episode {i+1}: reward={r:.4f}, length={l}")
    print(f"\nMean reward: {np.mean(episode_rewards):.4f} +/- {np.std(episode_rewards):.4f}")
    eval_env.close()

    check_env = TimeLimit(My4DOFTargetEnv(disturb_prob=0.3), max_episode_steps=1000)
    for ep in range(10):
        obs, info = check_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = check_env.step(action)
            done = terminated or truncated
        print(f"target_y={info['target_y']:.4f}, final com_y={info['com_y']:.4f}, "
              f"error={info['com_y']-info['target_y']:.4f}")
    check_env.close()