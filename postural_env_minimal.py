"""
MINIMAL, candidate-1-ONLY postural env. Every line commented for full
understanding. Just: 4-DOF model, target-reaching
reward, single-omega convex combination (effort vs tracking), real-torque
effort term, real joint limits, per-step success.

Goal of this file: be small enough that every design choice is visible. That's it. Literally stripped down version
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import mujoco
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium import spaces
import numpy as np

# ---------------------------------------------------------------------------
# SECTION 1: Model loading
# ---------------------------------------------------------------------------
# We reuse gymnasium's built-in InvertedPendulum-v5 loader machinery, but
# point it at OUR custom XML instead of the pendulum model that ships with
# gymnasium. 
XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ankle_hip_2x2dof.xml")
_base = gym.make("InvertedPendulum-v5", xml_file=XML_PATH).unwrapped
AnkleHipEnv = type(_base)

N_JOINTS = 4
JOINT_NAMES = ["ankle_eversion", "ankle_flexion", "hip_abduction", "hip_flexion"]

# ---------------------------------------------------------------------------
# SECTION 2: Constants
# ---------------------------------------------------------------------------
# Real, asymmetric joint limits in RADIANS, sourced from a real subject's
# URDF (subject3_single_leg_4dof.urdf)
JOINT_LOW = np.radians([-35.0, -50.0, -50.0, -30.0])
JOINT_HIGH = np.radians([35.0, 50.0, 30.0, 120.0])
# Failure triggers at 95% of the real limit, not 100% -- a small safety
# margin so failure is detected one step before actually slamming the
# physical joint stop (which can cause MuJoCo integration artifacts).
FAIL_MARGIN = 0.95

# Half-width (in meters) of the CoM-y target-sampling range during
# training. NOT the same as EPS_POS (the real 5mm success tolerance) --
# this only controls how far apart training targets are placed, and is a
# TUNED VALUE (0.047), not a physically-derived one. Chosen because 0.0566m
# (the real average target) used 90% of actuator torque ceiling and broke
# training; 0.047 is roughly 75% of ceiling and trains stably.
TARGET_RANGE = 0.047

# omega: the SINGLE scalar mixing weight between effort and tracking error
# in the convex combination EC_omega = omega*EF_u + (1-omega)*tracking_err. ONE weight only, no per-joint weights.
OMEGA = 0.2

# nd_cap: caps normalized tracking distance at 1.0 so it can't go negative
# via h = 1 - nd, and so it stays on the same [0,1] scale as EF_u (which
# is also constructed to max out near 1.0 -- see SECTION 5).
ND_CAP = 1.0

# Success reward. Paid EVERY STEP the state satisfies the success test
# (not once, not at episode end). This is intentional -- the model needs
# to be rewarded for STAYING at the target, not just touching it once.
# 2.0 chosen so it's comparable in magnitude to the ordinary h in [0,1] --
# an earlier attempt at 1000/step overwhelmed everything else.
SUCCESS_BONUS = 2.0

# Failure penalty: base -100, plus up to -400 more scaled by how far from
# target the failure happened (1-h). So failing right at the target costs
# -100, failing far away costs up to -500.
FAIL_BASE = -100.0
FAIL_SLOPE = -400.0

# Success test tolerances.
EPS_POS = 0.005   # meters, 5mm -- the REAL experimental target radius (confirmed value) POS = The position tolerance
EPS_VEL = 0.01    # m/s -- still an unsourced placeholder, maybe reasonable order of magnitude for "roughly stationary" VEL = The velocity tolerance (not sure tho)


class Candidate1Env(AnkleHipEnv):
    """
    Target-reaching postural control, mediolateral (CoM-y) axis only.
    No safety term (that's candidate 2/3 -- deliberately excluded here).
    """

    def __init__(self, target_range=TARGET_RANGE, fixed_target=None, omega=OMEGA, **kwargs):
        # Hand off model-loading/rendering/physics-stepping machinery to
        # the parent MujocoEnv class
        super().__init__(xml_file=XML_PATH, **kwargs)

        # Action space: 4 joints, each normalized to [-1, 1]. This is NOT
        # torque in Nm -- MuJoCo's actuator model converts this [-1,1]
        # "control signal" into real torque internally using each
        # actuator's `gear` value
        self.action_space = spaces.Box(-1.0, 1.0, (N_JOINTS,), np.float32)

        # Observation: [4 joint angles, 4 joint velocities, com_y/range,
        # target_y/range, (com_y-target_y)/range] = 11 dims.
        self.observation_space = spaces.Box(-np.inf, np.inf, (2 * N_JOINTS + 3,), np.float64)

        self.target_range = float(target_range)
        self.fixed_target = fixed_target   # None = random target each episode; else always this value
        self.omega = float(omega)

        self._max_steps = 1000
        self._current_step = 0
        self.target_y = 0.0
        self.prev_com_y = 0.0

        # Body IDs, looked up ONCE here (not every step) for performance
        # MuJoCo body lookups by name are string comparisons, expensive if
        # done every single simulation step.
        self.trunk_body_id = self.model.body("trunk").id
        self.root_body_id = self.model.body("foot").id

        # step_dt: real seconds per RL step. frame_skip = how many raw
        # physics substeps happen per RL step (MuJoCo's own timestep is
        # much finer than what the policy needs to act at).
        self.step_dt = self.model.opt.timestep * self.frame_skip

        # --- XCoM/capture-point constants, computed from the model itself,
        # never hand-typed. Not used for reward here (no safety term), but
        # IS used for the failure condition below (capture point exiting
        # base of support) -- kept because failure detection needs it even
        # in the no-safety candidate.
        d0 = mujoco.MjData(self.model)          # a throwaway data buffer, just for this one-time computation
        mujoco.mj_forward(self.model, d0)        # run one forward-kinematics pass to populate subtree_com etc.
        self.com_height = float(d0.subtree_com[self.root_body_id][2])   # z-height of CoM at the model's default pose
        self.omega0 = float(np.sqrt(9.81 / self.com_height))            # pendulum natural frequency, sqrt(g/h)
        # foot_geom's y-size = half the foot's mediolateral width = base of support half-width
        self.base_half_width = float(self.model.geom("foot_geom").size[1])

        # Real per-joint torque capacity (Nm), read directly from the
        # MuJoCo model's actuator_gear field. THIS is why action != torque:
        # real_torque_i = action_i * gear_i, and gear differs per joint
        # (documented as 30/50/50/75 in the handoff notes).
        self.joint_gears = self.model.actuator_gear[:, 0].copy()

        # effort_k: a scaling constant chosen so that IF all 4 actuators
        # were simultaneously at maximum torque (action=1 on all), EF_u
        # would equal exactly 1.0, matching the tracking-error term's
        # own [0,1] scale. Derivation: EF_u_max = K * sum(gear_i^2 * 1^2)
        # = K * sum(gear_i^2). Setting this to 1 gives K = 1/sum(gear_i^2).
        self.effort_k = 1.0 / float(np.sum(self.joint_gears ** 2))

        # Failure bounds = 95% of the real joint limits (see FAIL_MARGIN).
        self.fail_low = JOINT_LOW * FAIL_MARGIN
        self.fail_high = JOINT_HIGH * FAIL_MARGIN

    def _com_y(self):
        # subtree_com = center of mass of the body and everything below it
        # in the kinematic tree. Using the "foot" body's subtree_com gives
        # the WHOLE model's CoM (since foot is the root of the chain),
        # not just the foot segment's own CoM.
        return float(self.data.subtree_com[self.root_body_id][1])  # index 1 = y-axis (mediolateral)

    def _get_obs(self):
        q = self.data.qpos[:N_JOINTS].copy()   # current joint angles (radians)
        qd = self.data.qvel[:N_JOINTS].copy()  # current joint angular velocities (rad/s)
        com_y = self._com_y()
        # All position-like quantities are divided by target_range so the
        # policy's input features stay roughly O(1) in magnitude
        # unnormalized inputs (e.g. raw meters) can slow down NN training.
        return np.concatenate([
            q, qd,
            [com_y / self.target_range],
            [self.target_y / self.target_range],
            [(com_y - self.target_y) / self.target_range],
        ]).astype(np.float64)

    def reset(self, **kwargs):
        self._current_step = 0
        obs, info = super().reset(**kwargs)   # parent resets qpos/qvel to the model's defaults + small random noise

        # Dampen the parent's default reset noise to 10% of its usual size
        # MuJoCo's default reset randomization was too large for
        # this task (episodes were starting from already-unstable poses).
        self.data.qpos[:N_JOINTS] *= 0.1
        self.data.qvel[:N_JOINTS] *= 0.1
        mujoco.mj_forward(self.model, self.data)  # re-run kinematics so com_y etc. reflect the just-modified qpos

        # Pick a new random target each episode (or use a fixed one, for
        # evaluation grids that test specific target positions).
        self.target_y = (float(self.fixed_target) if self.fixed_target is not None
                          else float(self.np_random.uniform(-self.target_range, self.target_range)))
        self.prev_com_y = self._com_y()
        return self._get_obs(), info

    def step(self, action):
        # Advance the physics by `frame_skip` raw MuJoCo substeps, applying
        # `action` as the actuator control signal for all of them.
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        q = self.data.qpos[:N_JOINTS].copy()

        com_y = self._com_y()
        # Finite-difference velocity estimate: NOT read from a sensor, just
        # (current - previous) / dt. Simple but introduces one step of lag.
        com_y_dot = (com_y - self.prev_com_y) / self.step_dt
        self.prev_com_y = com_y

        # Capture point / XCoM: where the CoM would end up if it kept
        # moving at its current velocity and decelerated like an inverted
        # pendulum. Same quantity, different names in different fields
        # (Hof's "XCoM", Pratt's "capture point").
        xcom_y = com_y + com_y_dot / self.omega0

        # Failure = EITHER a joint exceeded 95% of its real limit, OR the
        # capture point exited the base of support (a "fall", by the
        # standard postural-control definition), OR the sim produced NaN/inf.
        failed = bool(not np.isfinite(obs).all()
                      or np.any(q < self.fail_low)
                      or np.any(q > self.fail_high)
                      or abs(xcom_y) > self.base_half_width)
        self._current_step += 1

        # Normalized tracking error: 0 = exactly on target, grows linearly,
        # capped at 1.0 by ND_CAP so it can't produce h < 0 (see below).
        norm_dist_raw = abs(com_y - self.target_y) / self.target_range
        nd = min(norm_dist_raw, ND_CAP)
        h = 1.0 - nd   # h=1 perfect tracking, h=0 at or beyond target_range error

        # Success: BOTH position AND velocity must be within tolerance --
        # i.e. the CoM must be AT the target AND roughly STATIONARY there,
        # not just passing through it. Checked every step; does not end
        # the episode (see SUCCESS_BONUS comment above).
        success = (abs(com_y - self.target_y) < EPS_POS
                   and abs(com_y_dot) < EPS_VEL
                   and not failed)

        if failed:
            reward = FAIL_BASE + FAIL_SLOPE * (1.0 - h)
            terminated = True
        elif success:
            reward = SUCCESS_BONUS
            terminated = False
        else:
            # ==========================================================
            # THE ACTION-VS-TORQUE DISTINCTION, made explicit:
            # `action` is MuJoCo's normalized control signal, [-1,1] per
            # joint, by construction of self.action_space above. It is
            # NOT physical torque. `self.joint_gears` converts it: real
            # torque (Nm) = action * gear, because that's literally how
            # MuJoCo's actuator model works internally (ctrl * gear =
            # applied generalized force, for a "motor" actuator type).
            # ==========================================================
            torque = action * self.joint_gears   # shape (4,), units: Nm
            ef_u = float(self.effort_k * np.sum(np.square(torque)))  # scaled to ~[0,1], see effort_k derivation

            # THE single-omega convex combination, exactly as required:
            # ec_omega in [0,1]-ish range, mixing effort cost and tracking
            # cost with ONE shared weight (no per-joint weighting).
            ec_omega = self.omega * ef_u + (1.0 - self.omega) * nd

            reward = h - ec_omega
            terminated = False

        info = {"com_y": com_y, "target_y": self.target_y, "h": h,
                "xcom_y": xcom_y, "com_y_dot": com_y_dot,
                "action": action.copy(), "torque": (action * self.joint_gears),
                "failed": failed, "success": success}
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False, info


if __name__ == "__main__":
    # Minimal smoke test: random policy, 5 steps, print action vs torque
    env = TimeLimit(Candidate1Env(), max_episode_steps=1000)
    obs, info = env.reset()
    for t in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"step {t}: action={np.round(action,3)}  torque_Nm={np.round(info['torque'],2)}  reward={reward:.3f}")
        if terminated or truncated:
            break
    env.close()
