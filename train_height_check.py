"""
train_height_check.py
=========================
Adds the upright/height success condition your supervisor suggested,
per Arditi's original h >= 0.99 criterion. Run this AFTER the A_SCALE
sweep (train_ascale_sweep.py), as a separate, isolated experiment -- only
ONE thing changes here (the success condition), so any behavioral
difference can be attributed cleanly to this change alone.

CHANGE FROM YOUR CONFIRMED CANDIDATE 2 CODE: success now additionally
requires com_z (current height) to be at least 90% of self.com_height
(the reference standing height computed in __init__ from a neutral
pose). This is the ONLY change from your working candidate2 file --
A_SCALE is left at 1.0 (untrained baseline value), not combined with
the sweep, to isolate this experiment's effect.

ASSUMPTION FLAGGED: uses self.com_height (already computed at
mj_forward on the neutral reset pose) as the "standing height"
reference. If your model has a more precise/direct standing-height
reference elsewhere, swap it in -- this is the most defensible existing
value in your code, but not verified against your specific MuJoCo model
beyond what's already in your pasted Candidate 2 file.

Usage:
    python train_height_check.py
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
A_SCALE = 1.0  # held fixed -- this experiment isolates the height-check change only

# NEW: upright/height success requirement, per supervisor suggestion.
HEIGHT_SUCCESS_FRACTION = 0.9  # com_z must be >= 90% of reference standing height


class Candidate2EnvHeightCheck(AnkleHipEnv):
    """Candidate 2, with success ALSO requiring com_z >= 0.9 * reference height."""

    def __init__(self, target_x_low=TARGET_X_LOW, target_x_high=TARGET_X_HIGH, fixed_target=None,
                 omega=OMEGA, shaping_weight=SHAPING_WEIGHT, nd_cap=ND_CAP, use_shaping=USE_SHAPING,
                 eps_pos=EPS_POS, eps_vel=EPS_VEL, safety_weight=SAFETY_WEIGHT,
                 com_y_weight=COM_Y_WEIGHT, stay_penalty_weight=STAY_PENALTY_WEIGHT,
                 disturb_prob=0.1, force_range=(0, 30), **kwargs):
        super().__init__(xml_file=XML_PATH, **kwargs)
        self.action_space = spaces.Box(-1.0, 1.0, (N_JOINTS,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (2 * N_JOINTS + 3,), np.float64)

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
        # NEW: reference standing height, used for the upright success check.
        self.reference_height = self.com_height

        self.fail_low = JOINT_LOW * FAIL_MARGIN
        self.fail_high = JOINT_HIGH * FAIL_MARGIN

    def _com_xyz(self):
        com_pos = self.data.subtree_com[self.root_body_id]
        return float(com_pos[0]), float(com_pos[1]), float(com_pos[2])

    def _get_obs(self):
        q = self.data.qpos[:N_JOINTS].copy()
        qd = self.data.qvel[:N_JOINTS].copy()
        com_x, com_y, com_z = self._com_xyz()
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
            com_x, com_y, com_z = self._com_xyz()
            error = target_start_x - com_x
            if abs(error) < 0.005:
                break
            self.data.qpos[1] = np.clip(self.data.qpos[1] + 0.04 * error,
                                        self.fail_low[1] * 0.75, self.fail_high[1] * 0.75)
            self.data.qpos[3] = np.clip(self.data.qpos[3] + 0.04 * error,
                                        self.fail_low[3] * 0.75, self.fail_high[3] * 0.75)
            mujoco.mj_forward(self.model, self.data)

        com_x, com_y, com_z = self._com_xyz()
        self.trial_start_x = com_x
        self.target_x = (float(self.fixed_target) if self.fixed_target is not None
                          else float(self.np_random.uniform(self.target_x_low, self.target_x_high)))
        com_x, com_y, com_z = self._com_xyz()
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

        com_x, com_y, com_z = self._com_xyz()
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

        # THE CHANGE: success now also requires being upright.
        is_upright = com_z >= HEIGHT_SUCCESS_FRACTION * self.reference_height
        success = (abs(com_x - self.target_x) < self.eps_pos
                   and abs(com_x_dot) < self.eps_vel
                   and is_upright
                   and not failed)

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

        reward = tracking - com_y_penalty + shaping + A_SCALE * (energy + safety)
        terminated = bool(failed)

        self._current_step += 1
        info = {"com_x": com_x, "com_y": com_y, "com_z": com_z, "target_x": self.target_x, "h": h,
                "xcom_x": xcom_x, "com_x_dot": com_x_dot, "failed": failed,
                "success": success, "is_upright": is_upright, "safety": safety,
                "com_y_penalty": com_y_penalty, "stay_penalty": stay_penalty}
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False, info


if __name__ == "__main__":
    log_dir = "./training_logs_candidate2_heightcheck_run2/"
    os.makedirs(log_dir, exist_ok=True)

    def make_env(rank):
        def _f():
            e = Candidate2EnvHeightCheck(disturb_prob=0.1, force_range=(0, 30))
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
    model.save("ppo_candidate2_heightcheck_run2")
    env.save("vecnormalize_candidate2_heightcheck_run2.pkl")
    env.close()
    print("\nNext: compare against your existing ppo_candidate2_ap_comy1_staypenalty_6")
    print("(the version WITHOUT the height check) using the same seeds -- if success")
    print("rate or final tracking error changes substantially, the original success")
    print("condition was indeed reachable while partially collapsed; if numbers barely")
    print("move, the joint-limit/capture-point terminations were already precluding it.")
