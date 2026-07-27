"""
Shared 4-DoF postural-control environment for candidates 1, 2 and 3.

One env, one reward core, the safety term swapped by a string flag. This
replaces three diverging copies of the same file -- a bug fixed in one copy
previously had to be re-fixed by hand in the others.

Changes from RL_4DoF_reward_candidate2_xcom.py, each justified:

  1. Observation now includes com_y and the tracking error, and every
     component is scaled to roughly unit magnitude. Previously target_y was
     +-0.05 while angles were +-0.5 and velocities order 1, so the target was
     the faintest signal in the vector; and com_y (the thing the reward is
     actually about) was not observed at all.
  2. norm_dist is used RAW for shaping (gradient survives far off-target,
     fixes the drift-into-limit failure) and BOUNDED for h (stops reward
     blow-up). The two roles were previously in conflict.
  3. Termination margin: joint limits for the failure test are 95% of the
     XML ranges. They were previously exactly equal to the ranges, which
     MuJoCo already hard-clamps, so "failure" only fired on numerical
     overshoot.
  4. Targets drawn from self.np_random, so seeding is honoured.
  5. XCoM constants recomputed at construction from the actual model rather
     than hard-coded, so they cannot silently drift out of date again.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "ankle_hip_2x2dof.xml")

_base = gym.make("InvertedPendulum-v5", xml_file=XML_PATH).unwrapped
AnkleHipEnv = type(_base)

JOINT_NAMES = ["ankle_eversion", "ankle_flexion", "hip_abduction", "hip_flexion"]
N_JOINTS = 4
JOINT_RANGE = np.array([0.3, 0.5, 0.5, 0.5])
FAIL_MARGIN = 0.95                    # terminate at 95% of range, not 100%

TARGET_RANGE = 0.03                   # see diagnose.py: 0.05 is 79% of the
                                      # actuator ceiling, 0.03 is 48%
OMEGA = 0.2                           # carried over from candidate 1
SHAPING_WEIGHT = 20.0                 # carried over from candidate 1
ND_CAP = 4.0                          # soft bound on norm_dist used for h

VEL_SCALE = 2.0                       # rough obs normalisation constants
SUCCESS_BONUS = 1000.0
FAIL_BASE = -100.0
FAIL_SLOPE = -400.0


import json


def load_run_config(run_dir):
    """Read what mode/safety/weight a run was trained with. Falls back to
    target/none for runs made before config.json was introduced."""
    path = os.path.join(run_dir, "config.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"mode": "target", "safety": "none", "weight": 0.0}


class PosturalEnv(AnkleHipEnv):
    """mode: 'target' (candidates 1/2/3) | 'preliminary' (baseline standing task)
    safety: 'none' (candidate 1) | 'xcom' (candidate 2) | 'capture' (candidate 3,
            capture point) | 'joint' (exploratory extra, not a real candidate --
            see _safety_term docstring)
            (safety is ignored when mode='preliminary')"""

    def __init__(self, mode="target", safety="none", safety_weight=0.0,
                 target_range=TARGET_RANGE, fixed_target=None, omega=OMEGA,
                 shaping_weight=SHAPING_WEIGHT, nd_cap=ND_CAP,
                 disturb_prob=0.0, force_range=(-20, 20), **kwargs):
        super().__init__(xml_file=XML_PATH, **kwargs)

        assert mode in ("target", "preliminary"), mode
        assert safety in ("none", "xcom", "capture", "joint"), safety
        self.mode = mode
        self.safety = safety
        self.safety_weight = float(safety_weight)
        self.target_range = float(target_range)
        self.fixed_target = fixed_target
        self.omega = float(omega)
        self.shaping_weight = float(shaping_weight)
        self.nd_cap = float(nd_cap)
        self.disturb_prob = disturb_prob
        self.force_range = force_range

        self.action_space = spaces.Box(-1.0, 1.0, (N_JOINTS,), np.float32)
        # preliminary has no target, so 2 fewer observation dims (no
        # target_y, no com-target error) than the target-reaching candidates
        obs_dim = (2 * N_JOINTS + 1) if mode == "preliminary" else (2 * N_JOINTS + 3)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float64)

        self._max_steps = 1000
        self._current_step = 0
        self.target_y = 0.0
        self.prev_norm_dist = 0.0
        self.prev_com_y = 0.0
        self.step_dt = self.model.opt.timestep * self.frame_skip

        self.trunk_body_id = self.model.body("trunk").id
        self.root_body_id = self.model.body("foot").id

        # XCoM constants derived from the model, never hard-coded.
        d0 = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, d0)
        self.com_height = float(d0.subtree_com[self.root_body_id][2])
        self.omega0 = float(np.sqrt(9.81 / self.com_height))
        self.base_half_width = float(
            self.model.geom("foot_geom").size[1])

        self.fail_low = -JOINT_RANGE * FAIL_MARGIN
        self.fail_high = JOINT_RANGE * FAIL_MARGIN

    # ------------------------------------------------------------------
    def _com_y(self):
        return float(self.data.subtree_com[self.root_body_id][1])

    def _get_obs(self):
        q = self.data.qpos[:N_JOINTS].copy()
        qd = self.data.qvel[:N_JOINTS].copy()
        if self.mode == "preliminary":
            return np.concatenate([q / JOINT_RANGE, qd / VEL_SCALE,
                                   [self._com_y() / self.target_range]]).astype(np.float64)
        com_y = self._com_y()
        return np.concatenate([
            q / JOINT_RANGE,
            qd / VEL_SCALE,
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

        if self.fixed_target is not None:
            self.target_y = float(self.fixed_target)
        else:
            self.target_y = float(self.np_random.uniform(
                -self.target_range, self.target_range))

        com_y = self._com_y()
        self.prev_norm_dist = abs(com_y - self.target_y) / self.target_range
        self.prev_com_y = com_y
        return self._get_obs(), info

    # ------------------------------------------------------------------
    def _safety_term(self, com_y, com_y_dot, q):
        """Returns a value <= 0. Each candidate's safety formulation."""
        if self.safety == "none":
            return 0.0
        if self.safety == "xcom":
            # Penalise only the velocity-extrapolation part of XCoM. The
            # absolute-position version penalised merely BEING at a nonzero
            # target more than moving toward it, making "stay at centre"
            # optimal regardless of target (verified arithmetically).
            instability = com_y_dot / self.omega0
            return -(instability / self.base_half_width) ** 2
        if self.safety == "capture":
            # Capture point: com_y + com_y_dot/omega0 -- mathematically the
            # same quantity as XCoM (Hof) and capture point (Pratt), just
            # named differently in different fields. Unlike the xcom branch
            # above, this checks the FULL point (position + velocity term)
            # against the actual base of support -- but as a THRESHOLD, not
            # a continuous quadratic pull to zero. Zero cost anywhere
            # inside the base of support (so reaching an off-center target
            # is free, as long as the capture point stays within the base);
            # cost only appears once it would actually exit the base and
            # risk a fall. This is what avoids the "stay at center" trap
            # that a continuous absolute-position penalty fell into earlier
            # (see the xcom history: candidate 2's first attempt).
            capture_point = com_y + com_y_dot / self.omega0
            excess = np.clip(abs(capture_point) - self.base_half_width, 0.0, None) \
                     / self.base_half_width
            return -(excess ** 2)
        if self.safety == "joint":
            # Proximity to joint limits. Zero in the interior of the
            # workspace, so unlike a velocity term it cannot compete with
            # the target term over the region where tracking happens.
            # NOTE: kept as an exploratory extra, not the primary candidate
            # 3 -- joint-limit terms are common in locomotion RL but not in
            # postural-control literature, which favours CoP/capture-point-
            # style base-of-support constraints instead. 'capture' above is
            # the one that matches the supervisor's actual request.
            frac = np.abs(q) / JOINT_RANGE
            excess = np.clip(frac - 0.7, 0.0, None) / 0.3
            return -float(np.sum(excess ** 2))
        raise ValueError(self.safety)

    def step(self, action):
        if self.disturb_prob > 0 and self.np_random.random() < self.disturb_prob:
            self.data.xfrc_applied[self.trunk_body_id, 0] = \
                self.np_random.uniform(*self.force_range)
        else:
            self.data.xfrc_applied[self.trunk_body_id, 0] = 0.0

        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        q = self.data.qpos[:N_JOINTS].copy()

        failed = bool(not np.isfinite(obs).all()
                      or np.any(q < self.fail_low)
                      or np.any(q > self.fail_high))
        self._current_step += 1

        com_y = self._com_y()
        com_y_dot = (com_y - self.prev_com_y) / self.step_dt
        self.prev_com_y = com_y
        xcom_y = com_y + com_y_dot / self.omega0

        if self.mode == "preliminary":
            # Baseline reward: penalize squared deviation from center plus
            # effort, matching the original preliminary script's structure
            # (position_term + effort_term + survive bonus). No target, no
            # shaping, no safety term -- this is the pre-Arditi baseline.
            terminated = failed
            if failed:
                reward = FAIL_BASE
            else:
                w1, w2 = 1.0, 0.01
                position_term = -w1 * (com_y ** 2)
                effort_term = -w2 * float(np.sum(np.square(action)))
                reward = position_term + effort_term + 0.1
            info = {"com_y": com_y, "target_y": 0.0, "h": 1.0 - abs(com_y) / self.target_range,
                   "xcom_y": xcom_y, "com_y_dot": com_y_dot,
                   "norm_dist_raw": abs(com_y) / self.target_range, "failed": failed}
            if self.render_mode == "human":
                self.render()
            return obs, reward, terminated, False, info

        # RAW for shaping: keeps a gradient pointing home even far off-target.
        norm_dist_raw = abs(com_y - self.target_y) / self.target_range
        shaping_bonus = self.prev_norm_dist - norm_dist_raw
        self.prev_norm_dist = norm_dist_raw

        # BOUNDED for h: prevents unbounded reward magnitudes.
        nd = min(norm_dist_raw, self.nd_cap)
        h = 1.0 - nd

        success = (self._current_step >= self._max_steps) and not failed
        if success:
            reward, terminated = SUCCESS_BONUS, True
        elif failed:
            reward = FAIL_BASE + FAIL_SLOPE * (1.0 - h)
            terminated = True
        else:
            ef_u = float(np.mean(np.square(action)))
            ec_omega = self.omega * ef_u + (1.0 - self.omega) * nd
            safety = self._safety_term(com_y, com_y_dot, q)
            reward = (h - ec_omega
                      + self.shaping_weight * shaping_bonus
                      + self.safety_weight * safety)
            terminated = False

        info = {"com_y": com_y, "target_y": self.target_y, "h": h,
                "xcom_y": xcom_y, "com_y_dot": com_y_dot,
                "norm_dist_raw": norm_dist_raw, "failed": failed}
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False, info