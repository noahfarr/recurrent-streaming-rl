from typing import Any

import jax.numpy as jnp
from flax import struct
from gymnax.wrappers.purerl import GymnaxWrapper
from streamlet.utils.typing import Array, EnvParams, Key


@struct.dataclass
class EpisodicLifeState:
    env_state: Any
    lives: Array

    def __getattr__(self, name):
        return getattr(self.env_state, name)

    @property
    def unwrapped(self):
        return getattr(self.env_state, "unwrapped", self.env_state)


class EpisodicLife(GymnaxWrapper):

    def reset(
        self, key: Key, params: EnvParams | None = None
    ) -> tuple[Array, EpisodicLifeState]:
        obs, env_state = self._env.reset(key, params)
        return obs, EpisodicLifeState(env_state, jnp.asarray(env_state.lives, jnp.int32))

    def step(
        self,
        key: Key,
        state: EpisodicLifeState,
        action: int | Array,
        params: EnvParams | None = None,
    ) -> tuple[Array, EpisodicLifeState, Array, Array, dict[str, Any]]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        lives = jnp.asarray(info["lives"], jnp.int32)
        life_lost = jnp.logical_and(lives < state.lives, lives > 0)
        return (
            obs,
            EpisodicLifeState(env_state, lives),
            reward,
            jnp.logical_or(done, life_lost),
            info,
        )
