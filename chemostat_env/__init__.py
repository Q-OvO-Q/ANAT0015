from gymnasium.envs.registration import register

register(
    id="ChemostatEnv-v0",
    entry_point="chemostat_env.envs:ChemostatEnv",
)
