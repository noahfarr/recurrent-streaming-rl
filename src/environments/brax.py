from enum import StrEnum

import jax.numpy as jnp
from streamlet.environments import environment

from src.environments.wrappers import MaskObservationWrapper


class Mode(StrEnum):
    FULL = "F"
    POSITION = "P"
    VELOCITY = "V"


masks = {
    "ant": {
        Mode.FULL: jnp.ones(27, dtype=jnp.bool),
        Mode.POSITION: jnp.zeros(27, dtype=jnp.bool).at[:13].set(True),
        Mode.VELOCITY: jnp.zeros(27, dtype=jnp.bool).at[13:].set(True),
    },
    "halfcheetah": {
        Mode.FULL: jnp.ones(17, dtype=jnp.bool),
        Mode.POSITION: jnp.zeros(17, dtype=jnp.bool)
        .at[jnp.array([0, 1, 2, 3, 8, 9, 10, 11, 12])]
        .set(True),
        Mode.VELOCITY: jnp.zeros(17, dtype=jnp.bool)
        .at[jnp.array([4, 5, 6, 7, 13, 14, 15, 16])]
        .set(True),
    },
    "hopper": {
        Mode.FULL: jnp.ones(11, dtype=jnp.bool),
        Mode.POSITION: jnp.zeros(11, dtype=jnp.bool).at[:5].set(True),
        Mode.VELOCITY: jnp.zeros(11, dtype=jnp.bool).at[5:].set(True),
    },
    "walker2d": {
        Mode.FULL: jnp.ones(17, dtype=jnp.bool),
        Mode.POSITION: jnp.zeros(17, dtype=jnp.bool).at[:8].set(True),
        Mode.VELOCITY: jnp.zeros(17, dtype=jnp.bool).at[8:].set(True),
    },
}


def make(env_id, mode=Mode.FULL, **kwargs):
    mode = Mode(mode)
    env, env_params = environment.make(f"brax::{env_id}", **kwargs)
    env = MaskObservationWrapper(env, masks[env_id][mode])
    return env, env_params
