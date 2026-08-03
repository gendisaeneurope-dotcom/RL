"""
Candidate F: action-based effort (from C) + effort always active + eps_vel=0.1
Standalone version
Reflects fixes from postural_env.py: torque-based effort, real joint
ranges, per-step (non-terminal) success. No safety/XCoM term (that's
candidate 2).
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
JOINT_NAMES = ["ankle_eversion", "ankle_flexion", "hip_abduction", "hip_flexion"]

# Real, asymmetric, URDF-derived joint limits (radians). NOT independently
# confirmed that URDF zero pose matches this sim's zero pose -- flagged
# assumption, not certainty.
JOINT_LOW = np.radians([-35.0, -50.0, -50.0, -30.0])
JOINT_HIGH = np.radians([35.0, 50.0, 30.0, 120.0])
FAIL_MARGIN = 0.95

TARGET_RANGE = 0.047       # training-scale denominator, NOT physically tied to eps_pos
OMEGA = 0.2                # convex-combination weight (effort vs tracking)
SHAPING_WEIGHT = 20.0      # off by default -- see USE_SHAPING below
ND_CAP = 1.0                # keeps tracking-error term on [0,1], matching EF_u

SUCCESS_BONUS = 2.0        # paid every step in-zone (NOT terminal)
FAIL_BASE = -100.0
FAIL_SLOPE = -400.0

EPS_POS = 0.005            # m, real experimental target radius (supervisor-confirmed)
EPS_VEL = 0.01             # m/s

USE_SHAPING = False        # paper's EC_omega has no shaping term; keep off unless justified


class Candidate1Env(AnkleHipEnv):
    """Target-reaching, single-omega convex combination, no safety term."""

    def __init__(self, target_range=TARGET_RANGE, fixed_target=None, omega=OMEGA,
                 shaping_weight=SHAPING_WEIGHT, nd_cap=ND_CAP, use_shaping=USE_SHAPING,
                 eps_pos=EPS_POS, eps_vel=EPS_VEL, disturb_prob=0.0,
                 force_range=(-20, 20), **kwargs):
        super().__init__(xml_file=XML_PATH, **kwargs)
        self.action_space = spaces.Box(-1.0, 1.0, (N_JOINTS,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (2 * N_JOINTS + 3,), np.float64)

        self.target_range = float(target_range)
        self.fixed_target = fixed_target
        self.omega = float(omega)
        self.shaping_weight = float(shaping_weight)
        self.nd_cap = float(nd_cap)
        self.use_shaping = bool(use_shaping)
        self.eps_pos = float(eps_pos)
        self.eps_vel = float(eps_vel)
        self.disturb_prob = disturb_prob
        self.force_range = force_range

        self._max_steps = 1000
        self._current_step = 0
        self.target_y = 0.0
        self.prev_norm_dist = 0.0
        self.prev_com_y = 0.0

        self.trunk_body_id = self.model.body("trunk").id
        self.root_body_id = self.model.body("foot").id
        self.step_dt = self.model.opt.timestep * self.frame_skip

        d0 = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, d0)
        self.com_height = float(d0.subtree_com[self.root_body_id][2])
        self.omega0 = float(np.sqrt(9.81 / self.com_height))
        self.base_half_width = float(self.model.geom("foot_geom").size[1])

        # Real per-joint torque limits (Nm) -- EF_u must be computed from
        # actual torque (action * gear), not raw action, since gear ratios
        # differ per joint (30/50/50/75) and action=1.0 means different Nm
        # on each joint.
        self.joint_gears = self.model.actuator_gear[:, 0].copy()
        self.effort_k = 1.0 / float(np.sum(self.joint_gears ** 2))

        self.fail_low = JOINT_LOW * FAIL_MARGIN
        self.fail_high = JOINT_HIGH * FAIL_MARGIN

    def _com_xy(self):
        com_pos = self.data.subtree_com[self.root_body_id]
        return float(com_pos[0]), float(com_pos[1])   # (x, y)

    def _get_obs(self):
        q = self.data.qpos[:N_JOINTS].copy()
        qd = self.data.qvel[:N_JOINTS].copy()
        com_x, com_y = self._com_xy()
        return np.concatenate([
            q, qd,
            [com_y / self.target_range],
            [self.target_y / self.target_range],
            [(com_y - self.target_y) / self.target_range],
        ]).astype(np.float64)

    def reset(self, **kwargs):
        self._current_step = 0
        obs, info = super().reset(**kwargs)
        self.data.qpos[:N_JOINTS] *= 0.1
        self.data.qvel[:N_JOINTS] *= 0.1
        mujoco.mj_forward(self.model, self.data)

        self.target_y = (float(self.fixed_target) if self.fixed_target is not None
                          else float(self.np_random.uniform(-self.target_range, self.target_range)))
        com_x, com_y = self._com_xy()
        self.prev_norm_dist = abs(com_y - self.target_y) / self.target_range
        self.prev_com_y = com_y
        return self._get_obs(), info

    def step(self, action):
        if self.disturb_prob > 0 and self.np_random.random() < self.disturb_prob:
            self.data.xfrc_applied[self.trunk_body_id, 1] = self.np_random.uniform(*self.force_range)
        else:
            self.data.xfrc_applied[self.trunk_body_id, 1] = 0.0


        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        q = self.data.qpos[:N_JOINTS].copy()

        com_x, com_y = self._com_xy()
        com_y_dot = (com_y - self.prev_com_y) / self.step_dt
        self.prev_com_y = com_y
        xcom_y = com_y + com_y_dot / self.omega0

        failed = bool(not np.isfinite(obs).all()
                    or np.any(q < self.fail_low)
                    or np.any(q > self.fail_high)
                    or abs(xcom_y) > self.base_half_width)
        self._current_step += 1

        norm_dist_raw = abs(com_y - self.target_y) / self.target_range
        shaping_bonus = self.prev_norm_dist - norm_dist_raw
        self.prev_norm_dist = norm_dist_raw

        nd = min(norm_dist_raw, self.nd_cap)
        h = 1.0 - nd

        success = (abs(com_y - self.target_y) < self.eps_pos
               and abs(com_y_dot) < self.eps_vel
               and not failed)

    # --- compute every term separately, unconditionally ---

    # energy: always active, every step, regardless of outcome
        ef_u = float(np.mean(np.square(action)))
        energy = -self.omega * ef_u

    # tracking: value depends on outcome, but always contributes
        if failed:
            tracking = FAIL_BASE + FAIL_SLOPE * (1.0 - h)
        elif success:
            tracking = SUCCESS_BONUS
        else:
            tracking = (1.0 - self.omega) * h

    # shaping: only meaningful while episode is ongoing
        shaping = (self.shaping_weight * shaping_bonus) if (self.use_shaping and not failed and not success) else 0.0

    # safety: placeholder for future safety term, kept explicit so nothing is silently dropped
        safety = 0.0

    # --- single combination line ---
        reward = energy + tracking + safety + shaping

        terminated = bool(failed)

        info = {"com_y": com_y, "com_x": com_x, "target_y": self.target_y, "h": h,
                "xcom_y": xcom_y, "com_y_dot": com_y_dot, "failed": failed, "success": success}
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False, info


if __name__ == "__main__":
    log_dir = "./training_logs_candidate1_F/"
    os.makedirs(log_dir, exist_ok=True)

    def make_env(rank):
        def _f():
            e = Candidate1Env()  
            e = TimeLimit(e, max_episode_steps=1000)
            e = Monitor(e, os.path.join(log_dir, f"monitor_{rank}"))
            return e
        return _f

    N_ENVS = 8  
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True)

    model = PPO("MlpPolicy", env, n_steps=2048, batch_size=256, ent_coef=0.01,
                learning_rate=3e-4, gamma=0.99, verbose=1)
    model.learn(total_timesteps=1_000_000)
    model.save("ppo_candidate1_F")
    env.save("vecnormalize_candidate1_F.pkl")
    env.close()
