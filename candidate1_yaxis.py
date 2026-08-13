"""
Candidate 1 -> axis-swapped, no safety term (baseline).

  tracking / target / success  -> com_y  (ML, task axis)
  off-axis stability           -> com_x  (AP, perturbed axis)
  perturbation                 -> axis 0 = com_x, orthogonal to the task

No safety term: this is the no-safety baseline. The off-axis stability
term on x remains, replacing the old com_y_penalty (which existed to
constrain the non-task axis -- now x).

Stay penalty removed: screening runs showed the 723-step pre-commit
window and target tracking are mutually exclusive in this setup.

BASE OF SUPPORT (XML box half-extents "0.15 0.08 0.05"):
    x (AP) half-length = 0.15 m
    y (ML) half-width  = 0.08 m

Usage:
    python candidate1_yaxis.py
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
EVERSION_J = 0

JOINT_LOW = np.radians([-35.0, -50.0, -50.0, -30.0])
JOINT_HIGH = np.radians([35.0, 50.0, 30.0, 120.0])
FAIL_MARGIN = 0.95

TARGET_SPAN = 0.5
COM_Y_PER_RAD = -0.31058 / np.radians(15.0)
TARGET_DISPLACEMENT = 0.107
START_NOISE = 0.005

OMEGA = 0.2
SHAPING_WEIGHT = 20.0
ND_CAP = 1.0

SUCCESS_BONUS = 6.0
FAIL_BASE = -100.0
FAIL_SLOPE = -150.0
A_SCALE = 1.0

EPS_POS = 0.005
EPS_VEL = 0.01
USE_SHAPING = False

# No safety term in Candidate 1. Off-axis weight matched to Candidate 2's
# swept value (0.30) so the only difference between candidates is the
# presence/formulation of the safety term.
OFF_AXIS_WEIGHT = 0.15


class Candidate1Env(AnkleHipEnv):
    """Tracking on com_y (ML); no safety term; off-axis stability on com_x."""

    def __init__(self, fixed_target=None, omega=OMEGA,
                 shaping_weight=SHAPING_WEIGHT, nd_cap=ND_CAP,
                 use_shaping=USE_SHAPING, eps_pos=EPS_POS, eps_vel=EPS_VEL,
                 off_axis_weight=OFF_AXIS_WEIGHT,
                 target_displacement=TARGET_DISPLACEMENT,
                 disturb_prob=0.1, force_range=(0, 30), **kwargs):
        super().__init__(xml_file=XML_PATH, **kwargs)
        self.action_space = spaces.Box(-1.0, 1.0, (N_JOINTS,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (2 * N_JOINTS + 3,), np.float64)

        self.fixed_target = fixed_target
        self.omega = float(omega)
        self.shaping_weight = float(shaping_weight)
        self.nd_cap = float(nd_cap)
        self.use_shaping = bool(use_shaping)
        self.eps_pos = float(eps_pos)
        self.eps_vel = float(eps_vel)
        self.off_axis_weight = float(off_axis_weight)
        self.target_displacement = float(target_displacement)
        self.disturb_prob = disturb_prob
        self.force_range = force_range

        self._max_steps = 1000
        self._current_step = 0
        self.target_y = 0.0
        self.prev_norm_dist = 0.0
        self.prev_com_x = 0.0
        self.prev_com_y = 0.0
        self.trial_start_y = 0.0
        self.safety_grace_steps = 30

        self.trunk_body_id = self.model.body("trunk").id
        self.root_body_id = self.model.body("trunk").id
        self.step_dt = self.model.opt.timestep * self.frame_skip

        d0 = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, d0)
        self.com_height = float(d0.subtree_com[self.root_body_id][2])
        self.omega0 = float(np.sqrt(9.81 / self.com_height))

        foot_size = self.model.geom("foot_geom").size
        self.base_half_length = float(foot_size[0])   # 0.15, AP
        self.base_half_width = float(foot_size[1])    # 0.08, ML

        self.joint_gears = self.model.actuator_gear[:, 0].copy()
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
            q, qd,
            [com_y / TARGET_SPAN],
            [self.target_y / TARGET_SPAN],
            [(com_y - self.target_y) / TARGET_SPAN],
        ]).astype(np.float64)

    def reset(self, **kwargs):
        self._current_step = 0
        obs, info = super().reset(**kwargs)
        self.data.qpos[:N_JOINTS] = 0.0
        self.data.qvel[:N_JOINTS] = 0.0
        mujoco.mj_forward(self.model, self.data)

        direction = 1.0 if self.np_random.random() < 0.5 else -1.0
        half = self.target_displacement / 2.0
        start_y = -direction * half + float(self.np_random.uniform(-START_NOISE, START_NOISE))
        goal_y = start_y + direction * self.target_displacement

        q_evers = np.clip(start_y / COM_Y_PER_RAD,
                          self.fail_low[EVERSION_J] * 0.75,
                          self.fail_high[EVERSION_J] * 0.75)
        self.data.qpos[EVERSION_J] = q_evers
        mujoco.mj_forward(self.model, self.data)

        for _ in range(20):
            com_x, com_y = self._com_xy()
            error = start_y - com_y
            if abs(error) < 0.002:
                break
            self.data.qpos[EVERSION_J] = np.clip(
                self.data.qpos[EVERSION_J] + 0.3 * error / COM_Y_PER_RAD,
                self.fail_low[EVERSION_J] * 0.75,
                self.fail_high[EVERSION_J] * 0.75)
            mujoco.mj_forward(self.model, self.data)

        com_x, com_y = self._com_xy()
        self.trial_start_y = com_y
        self.target_y = float(self.fixed_target) if self.fixed_target is not None else goal_y
        self.prev_norm_dist = abs(com_y - self.target_y) / TARGET_SPAN
        self.prev_com_x = com_x
        self.prev_com_y = com_y
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
        com_y_dot = (com_y - self.prev_com_y) / self.step_dt
        self.prev_com_x = com_x
        self.prev_com_y = com_y

        xcom_x = com_x + com_x_dot / self.omega0
        xcom_y = com_y + com_y_dot / self.omega0

        safety_active = self._current_step >= self.safety_grace_steps
        failed = bool(not np.isfinite(obs).all()
                      or np.any(q < self.fail_low) or np.any(q > self.fail_high)
                      or (safety_active and abs(xcom_x) > self.base_half_length)
                      or (safety_active and abs(xcom_y) > self.base_half_width))

        self._current_step += 1

        norm_dist_raw = abs(com_y - self.target_y) / TARGET_SPAN
        shaping_bonus = self.prev_norm_dist - norm_dist_raw
        self.prev_norm_dist = norm_dist_raw
        nd = min(norm_dist_raw, self.nd_cap)
        h = 1.0 - nd

        success = (abs(com_y - self.target_y) < self.eps_pos
                   and abs(com_y_dot) < self.eps_vel and not failed)

        ef_u = float(np.mean(np.square(action)))
        energy = -self.omega * ef_u

        if failed:
            tracking = FAIL_BASE + FAIL_SLOPE * (1.0 - h)
        elif success:
            tracking = SUCCESS_BONUS
        else:
            tracking = (1.0 - self.omega) * h

        shaping = (self.shaping_weight * shaping_bonus) if (self.use_shaping and not failed and not success) else 0.0

        # No safety term -- Candidate 1 is the no-safety baseline.
        safety = 0.0

        off_axis = self.off_axis_weight * (com_x / self.base_half_length) ** 2

        reward = tracking + shaping + A_SCALE * (energy + safety) - off_axis
        terminated = bool(failed)

        info = {"com_x": com_x, "com_y": com_y, "target_y": self.target_y,
                "h": h, "com_x_dot": com_x_dot, "com_y_dot": com_y_dot,
                "xcom_x": xcom_x, "xcom_y": xcom_y, "failed": failed,
                "success": success, "safety": safety, "off_axis": off_axis}
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False, info


if __name__ == "__main__":
    log_dir = "./training_logs_candidate1_yaxis_015/"
    os.makedirs(log_dir, exist_ok=True)

    def make_env(rank):
        def _f():
            e = Candidate1Env(disturb_prob=0.1, force_range=(0, 30))
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
    model.save("ppo_candidate1_yaxis_015")
    env.save("vecnormalize_candidate1_yaxis_015.pkl")
    env.close()
