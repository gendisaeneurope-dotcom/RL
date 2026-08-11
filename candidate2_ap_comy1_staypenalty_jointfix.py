
"""
Candidate 2 (stay-penalty variant, FIXED branch order, + JOINT-LIMIT-AWARE PENALTY):
energy + tracking (from Candidate F) + XCoM safety term + com_y stabilization
+ active stay-near-start penalty during the pre-commit window
+ NEW: joint-limit proximity penalty.

WHY THIS PATCH WAS ADDED (session finding): Candidates 1 and 3 both showed a
100%-consistent failure for target=-0.08 (every episode dies within ~150 steps),
while target=+0.08 always ran the full 1000 steps. Root cause identified: the
body's hip_flexion joint has an ASYMMETRIC range (-30 deg to +120 deg) baked into
the XML model. Reaching a negative target apparently requires a joint
configuration that sits closer to the tight (-30 deg) limit, and neither
existing safety term (XCoM velocity-based, or capture-point excess-based) has
ANY awareness of joint angles -- both only watch center-of-mass position and
velocity. So the policy gets no early warning before it slams into the tight
limit and fails.

THE FIX: add a smooth, anticipatory penalty that grows as ANY joint approaches
EITHER of its own limits (not the CoM), computed per-joint using each joint's
own actual low/high range. This gives the policy a reason to avoid joint
configurations near their edges well before the hard failure boundary,
independent of which direction the target is in.

Applied to Candidate 2 first, per instruction -- identical patch will be
ported to Candidates 1 and 3 afterward for a fair three-way retest.
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

# --- NEW: joint-limit-awareness parameters ---
# EDGE_ZONE: fraction of each joint's own range, measured from either end,
# where the penalty starts to kick in. 0.15 means the outer 15% of each
# joint's range (on both sides) is treated as a "danger zone."
JOINT_EDGE_ZONE = 0.15
# JOINT_LIMIT_WEIGHT: first-guess value, sized so the maximum possible
# per-step penalty (~1.0 at this weight, when a joint sits AT its hard
# limit) is comparable in magnitude to the energy and stay-penalty terms
# (~0.2-1.0 range) -- NOT swept yet, verify via joint-angle traces that
# this doesn't just make the policy avoid moving at all near either target.
JOINT_LIMIT_WEIGHT = 1.0


class Candidate2Env(AnkleHipEnv):
    """Candidate F's baseline (energy + tracking) plus XCoM safety + com_y stabilization
    + active stay-near-start penalty + NEW joint-limit-proximity penalty."""

    def __init__(self, target_x_low=TARGET_X_LOW, target_x_high=TARGET_X_HIGH, fixed_target=None, omega=OMEGA,
                 shaping_weight=SHAPING_WEIGHT, nd_cap=ND_CAP, use_shaping=USE_SHAPING,
                 eps_pos=EPS_POS, eps_vel=EPS_VEL, safety_weight=SAFETY_WEIGHT,
                 com_y_weight=COM_Y_WEIGHT, stay_penalty_weight=STAY_PENALTY_WEIGHT,
                 joint_edge_zone=JOINT_EDGE_ZONE, joint_limit_weight=JOINT_LIMIT_WEIGHT,
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
        self.joint_edge_zone = float(joint_edge_zone)
        self.joint_limit_weight = float(joint_limit_weight)
        self.disturb_prob = disturb_prob
        self.force_range = force_range

        self._max_steps = 1000
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

        self.joint_gears = self.model.actuator_gear[:, 0].copy()
        self.effort_k = 1.0 / float(np.sum(self.joint_gears ** 2))

        self.fail_low = JOINT_LOW * FAIL_MARGIN
        self.fail_high = JOINT_HIGH * FAIL_MARGIN

        # Precompute each joint's own range for the limit-proximity penalty.
        # Using the FULL (non-margin) range here, not fail_low/fail_high,
        # so the "danger zone" starts a bit before the hard failure boundary.
        self._joint_range = JOINT_HIGH - JOINT_LOW  # per-joint, shape (4,)

    def _com_xy(self):
        com_pos = self.data.subtree_com[self.root_body_id]
        return float(com_pos[0]), float(com_pos[1])

    def _joint_limit_penalty(self, q):
        """
        Computes a smooth, per-joint penalty that grows as any joint
        approaches EITHER of its own limits, using that joint's own range
        (so the asymmetric hip_flexion range is handled correctly -- this
        is normalized per-joint, not assuming symmetric limits).

        normalized_pos: 0.0 at JOINT_LOW, 1.0 at JOINT_HIGH, 0.5 at center.
        dist_to_nearest_edge: 0.5 at center, 0.0 at either hard limit.
        penalty: zero until within joint_edge_zone of an edge, then grows
                 quadratically as the joint gets closer to that edge.
        """
        normalized_pos = (q - JOINT_LOW) / self._joint_range
        dist_to_nearest_edge = np.minimum(normalized_pos, 1.0 - normalized_pos)
        violation = np.clip(self.joint_edge_zone - dist_to_nearest_edge, 0.0, None)
        # Normalize so violation=joint_edge_zone (i.e., AT the hard limit) gives penalty=1.0 per joint
        penalty_per_joint = (violation / self.joint_edge_zone) ** 2
        return float(np.sum(penalty_per_joint))

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
               and abs(com_x_dot) < self.eps_vel
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

        # --- NEW: joint-limit-proximity penalty, computed every step ---
        joint_limit_penalty = self._joint_limit_penalty(q)

        reward = (energy + tracking + safety - com_y_penalty + shaping
                  - self.joint_limit_weight * joint_limit_penalty)
        terminated = bool(failed)

        info = {"com_x": com_x, "com_y": com_y, "target_x": self.target_x, "h": h,
                "xcom_x": xcom_x, "com_x_dot": com_x_dot, "failed": failed,
                "success": success, "safety": safety, "com_y_penalty": com_y_penalty,
                "stay_penalty": stay_penalty, "joint_limit_penalty": joint_limit_penalty}

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False, info


if __name__ == "__main__":
    log_dir = "./training_logs_candidate2_ap_comy1_staypenalty_jointfix/"
    os.makedirs(log_dir, exist_ok=True)

    def make_env(rank):
        def _f():
            e = Candidate2Env(disturb_prob=0.1, force_range=(0, 30))
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
    model.save("ppo_candidate2_ap_comy1_staypenalty_jointfix")
    env.save("vecnormalize_candidate2_ap_comy1_staypenalty_jointfix.pkl")
    env.close()
