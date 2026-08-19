import jax
import jax.numpy as jnp
import numpy as np
from ale_py.vector_env import AtariVectorEnv
from flax import struct
from gymnax.environments import spaces

from src.utils.typing import Array


@struct.dataclass
class AleState:
    handle: Array


class AleEnvironment:
    def __init__(self, game, **kwargs):
        self._env = AtariVectorEnv(
            game=game,
            num_envs=1,
            reward_clipping=False,
            **kwargs,
        )
        handle, xla_reset, xla_step = self._env.xla()
        self._initial_handle = np.asarray(handle)
        self._xla_reset = xla_reset
        self._xla_step = xla_step
        self._num_actions = int(self._env.action_space.nvec[0])
        self._obs_shape = self._env.observation_space.shape[1:]

    @property
    def default_params(self):
        return None

    def reset(self, key, params=None):
        seed = jax.random.randint(key, (1,), 0, jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
        handle, (obs, info) = self._xla_reset(
            jnp.asarray(self._initial_handle), seed=seed
        )
        return obs[0], AleState(handle=handle)

    def step(self, key, state, action, params=None):
        handle, (obs, reward, terminated, truncated, info) = self._xla_step(
            state.handle, jnp.asarray(action, jnp.int32)[None]
        )
        done = jnp.logical_or(terminated[0], truncated[0])
        return (
            obs[0],
            AleState(handle=handle),
            jnp.asarray(reward[0], jnp.float32),
            done,
            {},
        )

    def action_space(self, params=None):
        return spaces.Discrete(self._num_actions)

    def observation_space(self, params=None):
        return spaces.Box(0, 255, self._obs_shape, dtype=jnp.uint8)


def make(env_id, **kwargs):
    return AleEnvironment(game=env_id, **kwargs), None
