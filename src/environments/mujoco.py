import jax.numpy as jnp
from streamlet.environments import environment

from src.environments.wrappers import MaskObservationWrapper

POSITION_DIMENSIONS = {
    "HalfCheetah-v4": 8,
    "Walker2d-v4": 8,
    "Hopper-v4": 5,
    "Ant-v4": 13,
    "Humanoid-v4": 22,
}


def make(env_id, mode="F", **kwargs):
    env, env_params = environment.make(f"gymnasium::{env_id}", **kwargs)
    if mode == "F":
        return env, env_params

    size = env.observation_space(env_params).shape[0]
    split = POSITION_DIMENSIONS[env_id]
    mask = jnp.zeros(size, dtype=bool)
    if mode == "P":
        mask = mask.at[:split].set(True)
    elif mode == "V":
        mask = mask.at[split:size].set(True)
    else:
        raise ValueError(f"unknown observation mode {mode}")
    return MaskObservationWrapper(env, mask), env_params
