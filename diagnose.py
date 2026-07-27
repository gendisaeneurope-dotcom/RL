"""
Pre-training gate. Runs in seconds. Nothing gets a training run until it
passes this.

Rationale: four reward variants were previously tested by 3M-timestep
training runs, each costing roughly an hour. Three of the four failures are
visible analytically, without training, from the reward landscape and the
model's actuator limits. Check those first.

Usage:  python diagnose.py
"""

import numpy as np
import mujoco

from postural_env import (PosturalEnv, XML_PATH, JOINT_RANGE, N_JOINTS,
                          TARGET_RANGE, OMEGA, SHAPING_WEIGHT, ND_CAP)

INTENDED_MASSES = {"leg": 11.0, "trunk": 35.0}
ANKLE_TORQUE_LIMIT = 30.0   # gear on ankle_eversion_motor, ctrlrange +-1


def check_masses(m):
    print("=" * 68)
    print("1. MASS CHECK")
    print("=" * 68)
    ok = True
    for name, want in INTENDED_MASSES.items():
        got = float(m.body_mass[m.body(name).id])
        flag = "OK" if abs(got - want) < 1e-6 else "MISMATCH"
        if flag != "OK":
            ok = False
        print(f"  {name:8s} intended {want:6.2f} kg   actual {got:6.2f} kg   {flag}")
    total = float(m.body_mass.sum())
    foot = float(m.body_mass[m.body("foot").id])
    print(f"  total {total:.2f} kg")
    print(f"  static foot is {100*foot/total:.1f}% of total mass and is INSIDE "
          f"subtree_com,\n    so com_y is diluted by that fraction "
          f"(known bias, document it).")
    if not ok:
        print("  >> FAIL: check compiler inertiafromgeom. Must be 'auto',"
              " not 'true'.")
    return ok


def check_actuation(m):
    print("=" * 68)
    print("2. ACTUATOR CEILING")
    print("=" * 68)
    d = mujoco.MjData(m)
    f = m.body("foot").id
    best = (0.0, 0.0, 0.0)
    for q in np.linspace(0, JOINT_RANGE[0], 800):
        d.qpos[:] = 0; d.qvel[:] = 0; d.qacc[:] = 0
        d.qpos[0] = -q
        mujoco.mj_forward(m, d)
        tau = abs(float(d.qfrc_bias[0] - d.qfrc_passive[0]))
        if tau <= ANKLE_TORQUE_LIMIT:
            best = (q, float(d.subtree_com[f][1]), tau)
    ceiling = best[1]
    util = 100 * TARGET_RANGE / ceiling if ceiling > 0 else float("inf")
    print(f"  max statically holdable com_y : {ceiling:.4f} m "
          f"(q_ev={best[0]:.4f}, tau={best[2]:.1f}/{ANKLE_TORQUE_LIMIT:.0f} Nm)")
    print(f"  TARGET_RANGE = {TARGET_RANGE:.3f} m -> {util:.0f}% of ceiling")
    if util > 60:
        print("  >> WARNING: edge targets leave little torque for disturbance"
              " rejection\n     or for a safety term that also wants control"
              " authority.")
    return util <= 60


def check_stability(m):
    print("=" * 68)
    print("3. PLANT TIMESCALE")
    print("=" * 68)
    d = mujoco.MjData(m); d.qpos[0] = 1e-4
    ts = []
    for _ in range(int(3.0 / m.opt.timestep)):
        d.ctrl[:] = 0
        mujoco.mj_step(m, d)
        ts.append(abs(float(d.qpos[0])))
    ts = np.array(ts)
    i1, i2 = np.argmax(ts > 1e-3), np.argmax(ts > 1e-2)
    tau_div = (i2 - i1) * m.opt.timestep / np.log(10) if i2 > i1 else np.inf
    dt = m.opt.timestep * 2
    print(f"  open-loop divergence time-constant : {tau_div:.3f} s")
    print(f"  control step                       : {dt:.4f} s")
    print(f"  -> {tau_div/dt:.0f} control steps per e-fold "
          f"({'controllable' if tau_div/dt > 10 else 'TIGHT'})")


def reward_landscape(safety, safety_weight, n_com=241, verbose=True):
    """For each target, is the per-step reward maximised AT the target?

    If the argmax is not at the target, the reward is wrong and no amount of
    ent_coef / network size / training budget will fix it. This is the check
    that would have caught the absolute-position XCoM failure before it cost
    a training run.
    """
    env = PosturalEnv(safety=safety, safety_weight=safety_weight)
    tr = env.target_range
    coms = np.linspace(-3 * tr, 3 * tr, n_com)
    targets = np.linspace(-tr, tr, 11)

    worst = 0.0
    rows = []
    for t in targets:
        rewards = []
        for cy in coms:
            nd_raw = abs(cy - t) / tr
            nd = min(nd_raw, ND_CAP)
            h = 1.0 - nd
            # evaluate the stationary case: zero velocity, zero action
            ec = OMEGA * 0.0 + (1.0 - OMEGA) * nd
            q = np.zeros(N_JOINTS)
            safe = env._safety_term(cy, 0.0, q)
            rewards.append(h - ec + safety_weight * safe)   # shaping = 0 at rest
        rewards = np.array(rewards)
        arg = coms[int(np.argmax(rewards))]
        err = abs(arg - t)
        worst = max(worst, err)
        rows.append((t, arg, err))
    env.close()

    if verbose:
        print(f"  {'target':>9} {'argmax com_y':>13} {'error(m)':>10}")
        for t, a, e in rows:
            print(f"  {t:>9.4f} {a:>13.4f} {e:>10.4f}")
    ok = worst < (2 * tr / (n_com - 1)) + 1e-9
    print(f"  worst argmax error = {worst:.4f} m  -> "
          f"{'PASS' if ok else 'FAIL (reward does not peak at the target)'}")
    return ok


if __name__ == "__main__":
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    r1 = check_masses(m)
    r2 = check_actuation(m)
    check_stability(m)

    print("=" * 68)
    print("4. REWARD LANDSCAPE (stationary, per-step reward argmax)")
    print("=" * 68)
    results = {}
    for name, safety, w in [("candidate 1 (none)",           "none",    0.0),
                            ("candidate 2 (xcom, w=1)",       "xcom",    1.0),
                            ("candidate 2 (xcom, w=5)",       "xcom",    5.0),
                            ("candidate 3 (capture, w=1)",    "capture", 1.0),
                            ("candidate 3 (capture, w=5)",    "capture", 5.0),
                            ("extra: joint-limit, w=1",       "joint",   1.0)]:
        print(f"\n-- {name}")
        results[name] = reward_landscape(safety, w, verbose=False)

    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)
    print(f"  masses correct        : {'PASS' if r1 else 'FAIL'}")
    print(f"  actuator headroom     : {'PASS' if r2 else 'WARN'}")
    for k, v in results.items():
        print(f"  {k:22s}: {'PASS' if v else 'FAIL'}")
    print("\nOnly launch training for configurations that PASS the landscape "
          "check.\nNote: this checks the STATIONARY reward only. A velocity-"
          "dependent safety\nterm can still distort the transient even when "
          "the stationary argmax is\ncorrect -- that is what the "
          "safety_weight sweep in train.py is for.")


def check_speed_budget(weights=(0.5, 1.0, 2.0, 5.0)):
    """The stationary landscape check cannot see the xcom term (it is zero at
    zero velocity). This is the check that can: how fast is the policy still
    ALLOWED to move before the safety penalty outweighs the shaping bonus?

    Compare the resulting traverse time against the plant's own open-loop
    divergence time-constant. If moving across the target range takes as long
    as the pendulum takes to fall, the safety weight is fighting the task.
    """
    env = PosturalEnv(safety="xcom", safety_weight=1.0)
    tr, dt, w0, bw = env.target_range, env.step_dt, env.omega0, env.base_half_width
    env.close()
    print("=" * 68)
    print("5. CANDIDATE 2 SPEED BUDGET (xcom safety term)")
    print("=" * 68)
    print(f"  shaping gain per step = {SHAPING_WEIGHT}*v*dt/tr = {SHAPING_WEIGHT*dt/tr:.3f}*v")
    print(f"  safety cost per step  = w*(v/(w0*bw))^2      = w*{1/(w0*bw)**2:.2f}*v^2")
    print(f"\n  {'weight':>7} {'max net-positive v':>19} {'time to cross range':>21}")
    for w in weights:
        v_zero = (SHAPING_WEIGHT * dt / tr) / (w / (w0 * bw) ** 2)
        t_cross = 2 * tr / v_zero
        print(f"  {w:>7.1f} {v_zero:>16.4f} m/s {t_cross:>18.3f} s")
    print(f"\n  open-loop divergence time-constant = 0.356 s")
    print("  A traverse time near or above that means the safety term is")
    print("  slowing the policy to the speed at which the plant falls over.")


def check_capture_speed_budget():
    """Capture point uses a THRESHOLD penalty (zero cost until the capture
    point exits the base of support), not a continuous quadratic like XCoM.
    So unlike XCoM, the free-movement speed here does NOT depend on the
    safety weight at all -- it's fixed by the base-of-support geometry.
    Confirms candidate 3 doesn't inherit XCoM's weight-sensitivity problem.
    """
    env = PosturalEnv(safety="capture", safety_weight=1.0)
    tr, w0, bw = env.target_range, env.omega0, env.base_half_width
    env.close()
    v_threshold = w0 * bw
    t_cross = 2 * tr / v_threshold
    print("=" * 68)
    print("6. CANDIDATE 3 SPEED BUDGET (capture point safety term)")
    print("=" * 68)
    print(f"  velocity threshold before ANY safety cost = {v_threshold:.4f} m/s")
    print(f"  (fixed by base_half_width * omega0 -- independent of safety_weight,")
    print(f"   unlike XCoM above)")
    print(f"  time to cross target range at that speed   = {t_cross:.3f} s")
    print(f"  open-loop divergence time-constant          = 0.356 s")
    ok = t_cross < 0.356
    print(f"  -> {'PASS' if ok else 'WARN'}: free-movement speed is "
          f"{'faster' if ok else 'NOT faster'} than the plant's own fall rate,")
    print(f"     so the safety term should not be forcing the policy to move "
          f"too slowly to balance.")


if __name__ == "__main__":
    check_speed_budget()
    check_capture_speed_budget()