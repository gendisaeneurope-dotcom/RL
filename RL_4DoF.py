import os

import mujoco

from RL_4DoF_target import My4DOFTargetEnv
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

# Real joint names from the verified URDF-derived model (subject3, scaled).
# Order: subtalar (ankle eversion/inversion) -> ankle (flexion) -> hip_rotation2 (abduction) -> hip_rotation1 (flexion)
JOINT_NAMES = ["ankle_eversion", "ankle_flexion", "hip_abduction", "hip_flexion"]
N_JOINTS = 4

# NOTE: unlike the old placeholder model, these ranges are NOT symmetric.
# Verified directly from the URDF via MuJoCo (see joint.range), in radians.
JOINT_LOW = np.array([-0.3, -0.5, -0.5, -0.5])
JOINT_HIGH = np.array([ 0.3,  0.5,  0.5,  0.5])


class My4DOFEnv(AnkleHipEnv):
    def __init__(self, disturb_prob=0.0, force_range=(-20, 20), omega=0.1, **kwargs):
        super().__init__(xml_file=xml_path, **kwargs)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(N_JOINTS,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 * N_JOINTS,), dtype=np.float64)
        self._current_step = 0
        self._max_steps = 1000
        self.disturb_prob = disturb_prob
        self.force_range = force_range
        self.omega = omega

        # Disturbance applied to the topmost consolidated segment (pelvis+torso, the
        # real analogue of the old "trunk" body). This is NOT the same body name as
        # before -- verify with a name lookup rather than assuming.
        self.trunk_body_id = self.model.body("trunk").id

        # "world" is the model root. The fixed foot geometry was merged into it by
        # MuJoCo's URDF compiler (since it's rigidly fixed, not a free body), so
        # subtree_com of "world" already gives the CoM of the entire moving chain.
        self.root_body_id = self.model.body("foot").id

    def reset(self, **kwargs):
        self._current_step = 0
        obs, info = super().reset(**kwargs)
        self.data.qpos[:N_JOINTS] *= 0.1
        self.data.qvel[:N_JOINTS] *= 0.1
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), info

    def _get_obs(self):
        angles = self.data.qpos[:N_JOINTS].copy()
        angvels = self.data.qvel[:N_JOINTS].copy()
        return np.concatenate([angles, angvels]).astype(np.float64)

    def step(self, action):
        if self.disturb_prob > 0 and np.random.rand() < self.disturb_prob:
            force = np.random.uniform(*self.force_range)
            self.data.xfrc_applied[self.trunk_body_id, 0] = force
        else:
            self.data.xfrc_applied[self.trunk_body_id, 0] = 0.0

        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        angles = observation[:N_JOINTS]

        # Asymmetric range check -- the old np.abs(angles) > JOINT_LIMITS check is
        # WRONG for this model (hip_c_rotation1 alone is -0.524 to +2.094, not
        # symmetric). Check lower/upper bounds separately.
        failed = bool(
            not np.isfinite(observation).all()
            or np.any(angles < JOINT_LOW)
            or np.any(angles > JOINT_HIGH)
        )

        self._current_step += 1
        terminated = failed

        if failed:
            print(f"Step {self._current_step}, angles(deg): {np.degrees(angles)}, disturb_prob: {self.disturb_prob}")

        # CoM of the entire moving chain (world subtree = everything, since foot
        # is rigidly fixed and merged into world already).
        com_pos = self.data.subtree_com[self.root_body_id][:2]
        base_center = np.array([0.0, 0.0])

        w1, w2 = 1.0, 0.01
        position_term = -w1 * np.sum((com_pos - base_center) ** 2)
        effort_term = -w2 * np.sum(np.square(action))

        # NOTE: the old ankle-hip coordination term (hardcoded 3:1 ratio) has been
        # REMOVED here, per supervisor feedback -- it was an unjustified constant
        # and effort_term already penalizes uncoordinated/wasteful action.
        reward = position_term + effort_term + 0.1

        info = {"reward_survive": reward}
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info


if __name__ == "__main__":
    log_dir = "./training_logs_4dof_v2/"
    os.makedirs(log_dir, exist_ok=True)

    env = TimeLimit(My4DOFEnv(render_mode="human"), max_episode_steps=1000)
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

    # -- Curriculum loop --
    stages = [
        {"disturb_prob": 0.0, "timesteps": 100_000},
        {"disturb_prob": 0.05, "timesteps": 150_000},
        {"disturb_prob": 0.15, "timesteps": 150_000},
        {"disturb_prob": 0.3, "timesteps": 100_000},
    ]

    for stage in stages:
        env.unwrapped.disturb_prob = stage["disturb_prob"]
        model.learn(total_timesteps=stage["timesteps"], reset_num_timesteps=False)
        print(f"Finished stage disturb_prob={stage['disturb_prob']}, "
              f"env reports: {env.unwrapped.disturb_prob}")
        model.save(f"ppo_ankle_hip_disturb_{stage['disturb_prob']}")

    model.save("ppo_ankle_hip")
    env.close()

    # -- Evaluation with rendering --
    model = PPO.load("ppo_ankle_hip")
    env = TimeLimit(My4DOFEnv(render_mode="human"), max_episode_steps=1000)
    obs, info = env.reset()

    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()

    # -- Quantitative evaluation (Monitor-wrapped to silence SB3's warning and get
    # accurate episode stats) --
    eval_env = TimeLimit(My4DOFEnv(disturb_prob=1.0, force_range=(-100, 100)), max_episode_steps=1000)
    eval_env = Monitor(eval_env)

    episode_rewards, episode_lengths = evaluate_policy(
        model, eval_env, n_eval_episodes=20, return_episode_rewards=True
    )
    for i, (r, l) in enumerate(zip(episode_rewards, episode_lengths)):
        print(f"Episode {i+1}: reward={r:.4f}, length={l}")

    print(f"\nMean reward: {np.mean(episode_rewards):.4f} +/- {np.std(episode_rewards):.4f}")
    eval_env.close()

    # -- Debug: zero-action survival + angle logging --
    debug_env = My4DOFEnv()
    obs, info = debug_env.reset()
    print("Initial angles (deg):", np.degrees(obs[:N_JOINTS]))

    angle_log = []
    obs, info = debug_env.reset()
    for step in range(500):
        action = np.zeros(N_JOINTS)
        obs, reward, terminated, truncated, info = debug_env.step(action)
        angles_deg = np.degrees(debug_env.data.qpos[:N_JOINTS])
        angle_log.append([step, *angles_deg])
        if terminated or truncated:
            print(f"Terminated at step {step}, angles: {angles_deg}")
            break
    else:
        print("Survived 500 steps with zero action")

    angle_log = np.array(angle_log)
    debug_env.close()
