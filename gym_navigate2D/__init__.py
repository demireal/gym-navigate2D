from gym.envs.registration import register

register(
    id='navigate2D-v0',
    entry_point='gym_navigate2D.envs:navigate2DEnv',
)
register(
    id='navigate2D-extrahard-v0',
    entry_point='gym_navigate2D.envs:navigate2DExtraHardEnv',
)