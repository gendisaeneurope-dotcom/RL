"""
train_ascale_sweep.py
=========================
Trains Candidate 2 at three A_SCALE values (1.0, 2.0, 4.0), per
supervisor suggestion: reward = tracking - com_y_penalty + shaping
                                + A_SCALE * (energy + safety)

A_SCALE=1.0 is the untrained baseline already defined in your candidate2
file (ppo_candidate2_ap_ascale1 was never actually run). This script
trains all three in one pass so they're directly comparable (same code,
same seeds, only A_SCALE differs).

Run this BEFORE the height-check experiment (Step 2) -- keep the two
changes separate so effects can be attributed cleanly.

Usage:
    python train_ascale_sweep.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import mujoco
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import numpy as np

XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ankle_hip_2x2dof.xml")
_base = gym.make("InvertedPendulum-v5", xml_file=XML_PATH).unwrapped
AnkleHipEnv = type(_base)

N_JOINTS = 4
JOINT_LOW = np.radians([-35.0, -50.0, -50.0, -30.0])
JOINT_HIGH = np.radians([35.0, 50.0, 30.0, 120.0])
FAIL_MARGIN = 0.95
TARGET_X_LOW, TARGET_X_HIGH = -0.1, 0.1
TARGET_SPAN = 0.5
OMEGA = 0.2
SHAPING_WEIGHT = 20.0
ND_CAP = 1.0
SUCCESS_BONUS = 6.0
FAIL_BASE = -100.0
FAIL_SLOPE = -150.0
TRACKING_DELAY_STEPS = 723
STAY_PENALTY_WEIGHT = 0.5
EPS_POS = 0.005
EPS_VEL = 0.01
USE_SHAPING = False
SAFETY_WEIGHT = 0.25
COM_Y_WEIGHT = 1.0

ASCALE_VALUES = [1.0, 2.0, 4.0]


class Candidate2EnvAScale(AnkleHipEnv):
    """Identical to your confirmed Candidate2Env, parameterized by a_scale."""

    def __init__(self, a_scale=1.0, target_x_low=TARGET_X_LOW, target_x_high=TARGET_X_HIGH,
                 fixed_target=None, omega=OMEGA, shaping_weight=SHAPING_WEIGHT, nd_cap=ND_CAP,
                 use_shaping=USE_SHAPING, eps_pos=EPS_POS, eps_vel=EPS_VEL,
                 safety_weight=SAFETY_WEIGHT, com_y_weight=COM_Y_WEIGHT,
                 stay_penalty_weight=STAY_PENALTY_WEIGHT, disturb_prob=0.1,
                 force_range=(0, 30), **kwargs):
        super().__init__(xml_file=XML_PATH, **kwargs)
        self.action_space = spaces.Box(-1.0, 1.0, (N_JOINTS,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (2 * N_JOINTS + 3,), np.float64)

        self.a_scale = float(a_scale)
        self.target_x_low = float(target_x_low)
        self.target_x_high = float(target_x_high)
        self.fixed_target = fixed_target
        self.omega = float(omega)
        self.shaping_weight = float(shaping_weight)
        self.nd_cap = float(nd_cap)
        self.use_shaping = bool(use_shaping)
        self.eps_pos = float(eps_pos)
        self.eps_vel = float(eps_vel)
        self.safety_weight = float(safety_weight)
        self.com_y_weight = float(com_y_weight)
        self.stay_penalty_weight = float(stay_penalty_weight)
        self.disturb_prob = disturb_prob
        self.force_range = force_range

        self._current_step = 0
        self.target_x = 0.0
        self.prev_norm_dist = 0.0
        self.prev_com_x = 0.0
        self.trial_start_x = 0.0
        self.safety_grace_steps = 30

        self.trunk_body_id = self.model.body("trunk").id
        self.root_body_id = self.model.body("trunk").id
        self.step_dt = self.model.opt.timestep * self.frame_skip

        d0 = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, d0)
        self.com_height = float(d0.subtree_com[self.root_body_id][2])
        self.omega0 = float(np.sqrt(9.81 / self.com_height))
        self.base_half_length = float(self.model.geom("foot_geom").size[0])

        self.fail_low = JOINT_LOW * FAIL_MARGIN
        self.fail_high = JOINT_HIGH * FAIL_MARGIN

    def _com_xy(self):
        com_pos = self.data.subtree_com[self.root_body_id]
        return float(com_pos[0]), float(com_pos[1])

    def _get_obs(self):
        q = self.data.qpos[:N_JOINTS].copy()
        qd = self.data.qvel[:N_JOINTS].copy()
        com_x, com_y = self._com_xy()
        return np.concatenate([
            q, qd, [com_x / TARGET_SPAN], [self.target_x / TARGET_SPAN],
            [(com_x - self.target_x) / TARGET_SPAN],
        ]).astype(np.float64)

    def reset(self, **kwargs):
        self._current_step = 0
        obs, info = super().reset(**kwargs)
        self.data.qpos[:N_JOINTS] *= 0.1
        self.data.qvel[:N_JOINTS] *= 0.1
        mujoco.mj_forward(self.model, self.data)

        target_start_x = float(self.np_random.uniform(self.target_x_low, self.target_x_high))
        for _ in range(150):
            com_x, com_y = self._com_xy()
            error = target_start_x - com_x
            if abs(error) < 0.005:
                break
            self.data.qpos[1] = np.clip(self.data.qpos[1] + 0.04 * error,
                                        self.fail_low[1] * 0.75, self.fail_high[1] * 0.75)
            self.data.qpos[3] = np.clip(self.data.qpos[3] + 0.04 * error,
                                        self.fail_low[3] * 0.75, self.fail_high[3] * 0.75)
            mujoco.mj_forward(self.model, self.data)

        com_x, com_y = self._com_xy()
        self.trial_start_x = com_x
        self.target_x = (float(self.fixed_target) if self.fixed_target is not None
                          else float(self.np_random.uniform(self.target_x_low, self.target_x_high)))
        com_x, com_y = self._com_xy()
        self.prev_norm_dist = abs(com_x - self.target_x) / TARGET_SPAN
        self.prev_com_x = com_x
        return self._get_obs(), info

    def step(self, action):
        if self.disturb_prob > 0 and self.np_random.random() < self.disturb_prob:
            self.data.xfrc_applied[self.trunk_body_id, 0] = self.np_random.uniform(*self.force_range)
        else:
            self.data.xfrc_applied[self.trunk_body_id, 0] = 0.0

        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        q = self.data.qpos[:N_JOINTS].copy()

        com_x, com_y = self._com_xy()
        com_x_dot = (com_x - self.prev_com_x) / self.step_dt
        self.prev_com_x = com_x
        xcom_x = com_x + com_x_dot / self.omega0

        safety_active = self._current_step >= self.safety_grace_steps
        failed = bool(not np.isfinite(obs).all()
                      or np.any(q < self.fail_low) or np.any(q > self.fail_high)
                      or (safety_active and abs(xcom_x) > self.base_half_length))

        norm_dist_raw = abs(com_x - self.target_x) / TARGET_SPAN
        shaping_bonus = self.prev_norm_dist - norm_dist_raw
        self.prev_norm_dist = norm_dist_raw
        nd = min(norm_dist_raw, self.nd_cap)
        h = 1.0 - nd

        success = (abs(com_x - self.target_x) < self.eps_pos
                   and abs(com_x_dot) < self.eps_vel and not failed)

        ef_u = float(np.mean(np.square(action)))
        energy = -self.omega * ef_u

        stay_penalty = 0.0
        if failed:
            tracking = FAIL_BASE + FAIL_SLOPE * (1.0 - h)
        elif self._current_step < TRACKING_DELAY_STEPS:
            stay_penalty = abs(com_x - self.trial_start_x) / TARGET_SPAN
            tracking = -self.stay_penalty_weight * stay_penalty
        elif success:
            tracking = SUCCESS_BONUS
        else:
            tracking = (1.0 - self.omega) * h

        shaping = (self.shaping_weight * shaping_bonus) if (self.use_shaping and not failed and not success) else 0.0

        instability = abs(com_x_dot / self.omega0) / self.base_half_length
        safety = -self.safety_weight * instability

        com_y_penalty = self.com_y_weight * (com_y / self.base_half_length) ** 2

        # THE CHANGE: energy + safety scaled together by a_scale.
        reward = tracking - com_y_penalty + shaping + self.a_scale * (energy + safety)
        terminated = bool(failed)

        self._current_step += 1
        info = {"com_x": com_x, "com_y": com_y, "target_x": self.target_x, "h": h,
                "xcom_x": xcom_x, "com_x_dot": com_x_dot, "failed": failed,
                "success": success, "safety": safety, "com_y_penalty": com_y_penalty,
                "stay_penalty": stay_penalty}
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False, info


def train_one(a_scale):
    tag = f"candidate2_ap_ascale{a_scale:g}".replace(".", "")
    log_dir = f"./training_logs_{tag}/"
    os.makedirs(log_dir, exist_ok=True)

    def make_env(rank):
        def _f():
            e = Candidate2EnvAScale(a_scale=a_scale, disturb_prob=0.1, force_range=(0, 30))
            e = TimeLimit(e, max_episode_steps=1000)
            e = Monitor(e, os.path.join(log_dir, f"monitor_{rank}"))
            return e
        return _f

    N_ENVS = 8
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    env = VecNormalize(env, norm_obs=False, norm_reward=False)

    model = PPO("MlpPolicy", env, n_steps=2048, batch_size=256, ent_coef=0.01,
                learning_rate=3e-4, gamma=0.99, verbose=1)
    model.learn(total_timesteps=3_000_000)
    model.save(f"ppo_{tag}")
    env.save(f"vecnormalize_{tag}.pkl")
    env.close()
    print(f"Saved ppo_{tag} / vecnormalize_{tag}.pkl")


if __name__ == "__main__":
    for a_scale in ASCALE_VALUES:
        print(f"\n=== Training A_SCALE={a_scale} ===")
        train_one(a_scale)
    print("\nAll three A_SCALE models trained. Next: run compare_resync_comy.py-style")
    print("comparison on each (ppo_candidate2_ap_ascale1/2/4) against the human data,")
    print("using the same FIXED_TARGET=0.1069 already established.")
