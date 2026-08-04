"""
Candidate 3: energy + tracking (from Candidate F) + capture-point safety term.

Same base as Candidate 2 (XCoM) -- only the safety term differs.

Capture point vs. XCoM, physically:
XCoM = com_y + com_y_dot / omega0            (extrapolated position)
Capture point safety = penalize only the EXCESS beyond the boundary,
                        not the raw magnitude everywhere.

This is structurally different from Candidate 2's safety term:
- XCoM's `instability` term has a non-zero gradient at ALL times (even
  when perfectly stable at center), since it just measures |com_y_dot/omega0|
  directly.
- Capture point's term is zero whenever the extrapolated position is safely
  within the boundary, and only grows once it exceeds it. This means the
  safety term stays completely silent unless the policy is actually near
  the edge (less likely to compete with tracking, since it has nothing
  to say about behavior deep inside the BoS).

Because of this, capture point is expected to be a "softer", less
intrusive safety term than XCoM
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

JOINT_LOW = np.radians([-35.0, -50.0, -50.0, -30.0])
JOINT_HIGH = np.radians([35.0, 50.0, 30.0, 120.0])
FAIL_MARGIN = 0.95

TARGET_X_LOW = -0.1
TARGET_X_HIGH = 0.1
TARGET_SPAN = 0.5

OMEGA = 0.2
SHAPING_WEIGHT = 20.0
ND_CAP = 1.0

SUCCESS_BONUS = 2.0
FAIL_BASE = -100.0
FAIL_SLOPE = -150.0

EPS_POS = 0.005
EPS_VEL = 0.01

USE_SHAPING = False

# Capture-point safety weight. Same starting point as XCoM's for a fair
# comparison if this term is structurally "softer" as expected, it may
# tolerate a higher weight before causing fixed-point issues. Tune only
# after confirming target-tracking still works at weight=1.0.
SAFETY_WEIGHT = 1.0


class Candidate3Env(AnkleHipEnv):
    """Candidate F's baseline (energy + tracking) plus a capture-point safety term."""

    def __init__(self, target_x_low=TARGET_X_LOW, target_x_high=TARGET_X_HIGH, fixed_target=None, omega=OMEGA,
                 shaping_weight=SHAPING_WEIGHT, nd_cap=ND_CAP, use_shaping=USE_SHAPING,
                 eps_pos=EPS_POS, eps_vel=EPS_VEL, safety_weight=SAFETY_WEIGHT,
                 disturb_prob=0.0, force_range=(-20, 20), **kwargs):
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
        self.disturb_prob = disturb_prob
        self.force_range = force_range

        self._max_steps = 1000
        self._current_step = 0
        self.target_x = 0.0
        self.prev_norm_dist = 0.0
        self.prev_com_x = 0.0
        self.safety_grace_steps = 30

        self.trunk_body_id = self.model.body("trunk").id
        self.root_body_id = self.model.body("trunk").id
        self.step_dt = self.model.opt.timestep * self.frame_skip

        d0 = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, d0)
        self.com_height = float(d0.subtree_com[self.root_body_id][2])
        self.omega0 = float(np.sqrt(9.81 / self.com_height))
        self.base_half_length = float(self.model.geom("foot_geom").size[0])

        self.joint_gears = self.model.actuator_gear[:, 0].copy()
        self.effort_k = 1.0 / float(np.sum(self.joint_gears ** 2))

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
            [com_x / TARGET_SPAN],
            [self.target_x / TARGET_SPAN],
            [(com_x - self.target_x) / TARGET_SPAN],
        ]).astype(np.float64)

    def reset(self, **kwargs):
        self._current_step = 0
        obs, info = super().reset(**kwargs)
        self.data.qpos[:N_JOINTS] *= 0.1
        self.data.qvel[:N_JOINTS] *= 0.1
        mujoco.mj_forward(self.model, self.data)

        target_start_x = 0.45
        for _ in range(150):
            com_x, _ = self._com_xy()
            error = target_start_x - com_x
            if abs(error) < 0.005:
                break
            self.data.qpos[1] = np.clip(self.data.qpos[1] + 0.04 * error,
                                        self.fail_low[1] * 0.75, self.fail_high[1] * 0.75)
            self.data.qpos[3] = np.clip(self.data.qpos[3] + 0.04 * error,
                                        self.fail_low[3] * 0.75, self.fail_high[3] * 0.75)
            mujoco.mj_forward(self.model, self.data)


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
                   and abs(com_x_dot) < self.eps_vel
                   and not failed)

        # --- every term computed separately, unconditionally ---

        ef_u = float(np.mean(np.square(action)))
        energy = -self.omega * ef_u

        if failed:
            tracking = FAIL_BASE + FAIL_SLOPE * (1.0 - h)
        elif success:
            tracking = SUCCESS_BONUS
        else:
            tracking = (1.0 - self.omega) * h

        shaping = (self.shaping_weight * shaping_bonus) if (self.use_shaping and not failed and not success) else 0.0

        # safety: CAPTURE POINT. capture_point = com_x + com_x_dot/omega0
        # (identical math to XCoM), but the safety term itself only
        # penalizes the EXCESS beyond the boundary via a clip, not the raw
        # magnitude. This is the structural difference from Candidate 2.
        capture_point = com_x + com_x_dot / self.omega0
        excess = np.clip(abs(capture_point) - self.base_half_length, 0.0, None) / self.base_half_length
        safety = -self.safety_weight * (excess ** 2)

        # --- single combination line ---
        reward = energy + tracking + safety + shaping
        terminated = bool(failed)

        info = {"com_x": com_x, "com_y": com_y, "target_x": self.target_x, "h": h,
                "xcom_x": xcom_x, "com_x_dot": com_x_dot, "failed": failed,
                "success": success, "safety": safety, "capture_point": capture_point}
        
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False, info


if __name__ == "__main__":
    log_dir = "./training_logs_candidate3_ap/"
    os.makedirs(log_dir, exist_ok=True)

    def make_env(rank):
        def _f():
            e = Candidate3Env()
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
    model.save("ppo_candidate3_ap")
    env.save("vecnormalize_candidate3_ap.pkl")
    env.close()