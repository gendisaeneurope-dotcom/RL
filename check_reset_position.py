import candidate3_ap as module
EnvClass = module.Candidate3Env

for target in [-0.08, -0.04, 0.0, 0.04, 0.08]:
    env = EnvClass(fixed_target=target)
    obs, info = env.reset()
    print(f"target={target}  initial com_xy={env._com_xy()}")