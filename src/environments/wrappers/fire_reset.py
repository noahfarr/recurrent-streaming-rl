from typing import Any

import jax.numpy as jnp
from flax import struct
from gymnax.wrappers.purerl import GymnaxWrapper
from streamlet.utils.typing import Array, EnvParams, Key


@struct.dataclass
class FireResetState:
    env_state: Any
    needs_fire: Array

    def __getattr__(self, name):
        return getattr(self.env_state, name)

    @property
    def unwrapped(self):
        return getattr(self.env_state, "unwrapped", self.env_state)


class FireReset(GymnaxWrapper):

    def __init__(self, env, fire_action: int = 1):
        super().__init__(env)
        self.fire_action = fire_action

    def reset(
        self, key: Key, params: EnvParams | None = None
    ) -> tuple[Array, FireResetState]:
        obs, env_state = self._env.reset(key, params)
        return obs, FireResetState(env_state, jnp.bool_(True))

    def step(
        self,
        key: Key,
        state: FireResetState,
        action: int | Array,
        params: EnvParams | None = None,
    ) -> tuple[Array, FireResetState, Array, Array, dict[str, Any]]:
        action = jnp.where(state.needs_fire, self.fire_action, action)
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        return obs, FireResetState(env_state, done), reward, done, info
