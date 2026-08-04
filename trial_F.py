from candidate2_ap import Candidate2Env
env = Candidate2Env()
obs, info = env.reset()
print("base_half_length:", env.base_half_length)
com_x, _ = env._com_xy()
print("reset com_x:", com_x, "vs boundary:", env.base_half_length)