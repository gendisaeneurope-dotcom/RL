import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import mujoco
import mujoco.viewer
import numpy as np

xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ankle_knee_hip_trunk.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Basic structural checks first
print("Number of joints:", model.njnt)
print("Joint names:", [model.joint(i).name for i in range(model.njnt)])
print("Number of bodies:", model.nbody)
print("Body names:", [model.body(i).name for i in range(model.nbody)])
print("Number of actuators:", model.nu)
print("qpos size:", model.nq, "| qvel size:", model.nv)

# Sanity check: should be exactly 4 hinge joints (ankle, knee, hip, trunk)
assert model.njnt == 4, f"Expected 4 joints, got {model.njnt}"

# Visualize and step through manually
with mujoco.viewer.launch_passive(model, data) as viewer:
    for step in range(2000):
        # Apply small random actions to see the chain move/react
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
        viewer.sync()
        if step % 200 == 0:
            print(f"step {step}: qpos={np.round(data.qpos[:4], 3)}")