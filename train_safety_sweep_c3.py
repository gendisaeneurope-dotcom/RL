"""
train_safety_sweep_c3.py  ->  CANDIDATE 3 (capture-point safety)
================================================================
Axis-swapped: task on com_y (ML), safety on com_x (AP).

Differs from Candidate 2 only in the safety formulation:
    C2:  -w * |com_x_dot / omega0| / L_AP           velocity, continuous
    C3:  -w * [max(0, |cp| - L_AP) / L_AP]^2        position, deadzone


Usage:
    python train_safety_sweep_c3.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import mujoco
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
import numpy as np

XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ankle_hip_2x2dof.xml")
_base = gym.make("InvertedPendulum-v5", xml_file=XML_PATH).unwrapped
AnkleHipEnv = type(_base)
 
N_JOINTS = 4
EVERSION_J = 0
 
JOINT_LOW = np.radians([-35.0, -50.0, -50.0, -30.0])
JOINT_HIGH = np.radians([35.0, 50.0, 30.0, 120.0])
FAIL_MARGIN = 0.95
 
TARGET_SPAN = 0.5
COM_Y_PER_RAD = -0.31058 / np.radians(15.0)
TARGET_DISPLACEMENT = 0.107   # was 0.08 -- must match the corrected 0.15m geometry
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
 
TRAIN_STEPS = 3_000_000   # the confirmed-stable check was run at 3M
                          
 
# ---------------------------------------------------------------------
RUN_MODE = "seeds"           # "seeds" or "sweep"
 
SEED_CHECK_WEIGHT = 0.50     # single-run check showed 4/5 good, 1 miss on
                              # the positive target (ep2, err 0.0272) --
                              # this seed run determines if that's real or
                              # this specific seed's variance
SEEDS = [0, 1, 2]
 
SAFETY_VALUES = [0.05, 0.15, 0.25, 0.50]   # used in "sweep" mode
SWEEP_SEED = 0
# ---------------------------------------------------------------------
 
 
class Candidate3EnvY(AnkleHipEnv):
    """Axis-swapped Candidate 3, parameterised by safety_weight."""

    def __init__(self, safety_weight=0.50, off_axis_weight=None,
                 fixed_target=None, omega=OMEGA, shaping_weight=SHAPING_WEIGHT,
                 nd_cap=ND_CAP, use_shaping=USE_SHAPING, eps_pos=EPS_POS,
                 eps_vel=EPS_VEL, target_displacement=TARGET_DISPLACEMENT,
                 disturb_prob=0.1, force_range=(0, 30), **kwargs):
        super().__init__(xml_file=XML_PATH, **kwargs)
        self.action_space = spaces.Box(-1.0, 1.0, (N_JOINTS,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (2 * N_JOINTS + 3,), np.float64)

        self.safety_weight = float(safety_weight)
        self.off_axis_weight = float(off_axis_weight) if off_axis_weight is not None \
            else float(safety_weight) * 2.0
        self.fixed_target = fixed_target
        self.omega = float(omega)
        self.shaping_weight = float(shaping_weight)
        self.nd_cap = float(nd_cap)
        self.use_shaping = bool(use_shaping)
        self.eps_pos = float(eps_pos)
        self.eps_vel = float(eps_vel)
        self.target_displacement = float(target_displacement)
        self.disturb_prob = disturb_prob
        self.force_range = force_range

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
        self.base_half_length = float(foot_size[0])   # AP
        self.base_half_width = float(foot_size[1])    # ML

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

        # CAPTURE POINT on the AP axis -- the only difference from C2.
        capture_point = com_x + com_x_dot / self.omega0
        excess = np.clip(abs(capture_point) - self.base_half_length, 0.0, None) / self.base_half_length
        safety = -self.safety_weight * (excess ** 2)

        off_axis = self.off_axis_weight * (com_x / self.base_half_length) ** 2

        reward = tracking + shaping + A_SCALE * (energy + safety) - off_axis
        terminated = bool(failed)

        info = {"com_x": com_x, "com_y": com_y, "target_y": self.target_y,
                "h": h, "com_x_dot": com_x_dot, "com_y_dot": com_y_dot,
                "xcom_x": xcom_x, "xcom_y": xcom_y, "capture_point": capture_point,
                "failed": failed, "success": success, "safety": safety,
                "off_axis": off_axis}
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False, info


def train_one(safety_weight, seed):
    tag = f"c3_yaxis_sw{safety_weight:g}_s{seed}".replace(".", "")
    log_dir = f"./training_logs_{tag}/"
    os.makedirs(log_dir, exist_ok=True)

    def make_env(rank):
        def _f():
            e = Candidate3EnvY(safety_weight=safety_weight,
                               disturb_prob=0.1, force_range=(0, 30))
            e = TimeLimit(e, max_episode_steps=1000)
            e = Monitor(e, os.path.join(log_dir, f"monitor_{rank}"))
            return e
        return _f

    N_ENVS = 8
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    env.seed(int(seed))
    env = VecNormalize(env, norm_obs=False, norm_reward=False)

    model = PPO("MlpPolicy", env, n_steps=2048, batch_size=256, ent_coef=0.01,
                learning_rate=3e-4, gamma=0.99, verbose=1, seed=int(seed))
    model.learn(total_timesteps=TRAIN_STEPS)
    model.save(f"ppo_{tag}")
    env.save(f"vecnormalize_{tag}.pkl")
    env.close()
    return tag


def evaluate(tag, safety_weight, n_eps=20):
    model = PPO.load(f"ppo_{tag}")
    errors, lengths, ap_ptp, wrong_side = [], [], [], 0
    for ep in range(n_eps):
        venv = DummyVecEnv([lambda: TimeLimit(
            Candidate3EnvY(safety_weight=safety_weight), max_episode_steps=1000)])
        venv = VecNormalize.load(f"vecnormalize_{tag}.pkl", venv)
        venv.training = False
        venv.norm_reward = False
        venv.seed(ep)
        obs = venv.reset()
        done, n = False, 0
        ap_trace = []
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, d, info = venv.step(a)
            done = bool(d[0]); n += 1
            ap_trace.append(float(info[0]["com_x"]))
        i = info[0]
        errors.append(abs(i["com_y"] - i["target_y"]))
        lengths.append(n)
        ap_ptp.append(float(np.ptp(ap_trace)))
        if np.sign(i["com_y"]) != np.sign(i["target_y"]):
            wrong_side += 1
        venv.close()
    return (float(np.mean(errors)), float(np.mean(lengths)),
            float(np.mean(ap_ptp)), wrong_side,
            sum(1 for e in errors if e < EPS_POS))


if __name__ == "__main__":
    results = []

    if RUN_MODE == "seeds":
        print(f"C3 SEED REPLICATION: safety={SEED_CHECK_WEIGHT}, seeds={SEEDS}")
        for sd in SEEDS:
            print(f"\n=== C3 safety={SEED_CHECK_WEIGHT}, seed={sd} ===")
            tag = train_one(SEED_CHECK_WEIGHT, seed=sd)
            r = evaluate(tag, SEED_CHECK_WEIGHT)
            results.append((f"seed {sd}", SEED_CHECK_WEIGHT) + r)
            print(f"  err {r[0]:.4f} | len {r[1]:.0f} | AP p-p {r[2]:.4f} "
                  f"| wrong {r[3]}/20 | hit {r[4]}/20")
    else:
        print(f"C3 WEIGHT SWEEP: values={SAFETY_VALUES}, seed={SWEEP_SEED}")
        for sw in SAFETY_VALUES:
            print(f"\n=== C3 safety={sw} (off_axis={sw*2:g}), seed={SWEEP_SEED} ===")
            tag = train_one(sw, seed=SWEEP_SEED)
            r = evaluate(tag, sw)
            results.append((f"sw {sw:g}", sw) + r)
            print(f"  err {r[0]:.4f} | len {r[1]:.0f} | AP p-p {r[2]:.4f} "
                  f"| wrong {r[3]}/20 | hit {r[4]}/20")

    print("\n" + "=" * 84)
    print(f"CANDIDATE 3 (capture-point safety) -- mode: {RUN_MODE}")
    print(f"{'run':>10} {'safety':>8} {'off_axis':>9} {'mean_err':>10} "
          f"{'len':>6} {'AP p-p':>9} {'wrong':>8} {'hit':>7}")
    print("-" * 84)
    for label, sw, err, ln, ap, wrong, hit in results:
        print(f"{label:>10} {sw:>8.2f} {sw*2:>9.2f} {err:>10.4f} "
              f"{ln:>6.0f} {ap:>9.4f} {wrong:>6}/20 {hit:>5}/20")
    print("=" * 84)
    print("\nHuman AP peak-to-peak reference: 0.0077 m")
    if RUN_MODE == "seeds":
        print("\nC1 seed check at off_axis=0.15 gave err 0.0172 / 0.0008 / 0.0305")
        print("(hit 10/20, 20/20, 10/20) -- variance dominated. If C3 shows")
        print("the same spread, no weight can be claimed as selected.")