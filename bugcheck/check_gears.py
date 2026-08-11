
from candidate2_ap_comy1_staypenalty import Candidate2Env
env = Candidate2Env()
obs, info = env.reset()
print("Observation shape:", obs.shape, "-- should be (11,)")
print("Observation values:", obs)
print("Action space:", env.action_space, "-- should be Box(-1,1,(4,))")
print("Joint gears (torque = action * gear):", env.joint_gears)

action = env.action_space.sample()
print("Sample action:", action)
print("Resulting torque (action * gear):", action * env.joint_gears)