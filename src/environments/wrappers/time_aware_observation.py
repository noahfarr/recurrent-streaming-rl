from typing import Any

import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces
from gymnax.wrappers.purerl import GymnaxWrapper
from streamlet.utils.typing import Array, EnvParams, Key


@struct.dataclass
class TimeAwareObservationState:
    env_state: Any
    time: Array

    def __getattr__(self, name):
        return getattr(self.env_state, name)

    @property
    def unwrapped(self):
        return getattr(self.env_state, "unwrapped", self.env_state)


class TimeAwareObservationWrapper(GymnaxWrapper):

    def __init__(self, env, time_limit: int):
        super().__init__(env)
        self.time_limit = time_limit

    def observation_space(self, params: EnvParams | None = None) -> spaces.Box:
        space = self._env.observation_space(params)
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(space.shape[0] + 1,),
            dtype=space.dtype,
        )

    def _append_time(self, obs: Array, time: Array) -> Array:
        feature = time / self.time_limit - 0.5
        return jnp.concatenate([obs, feature[None].astype(obs.dtype)])

    def reset(
        self, key: Key, params: EnvParams | None = None
    ) -> tuple[Array, TimeAwareObservationState]:
        obs, env_state = self._env.reset(key, params)
        time = jnp.int32(0)
        return self._append_time(obs, time), TimeAwareObservationState(env_state, time)

    def step(
        self,
        key: Key,
        state: TimeAwareObservationState,
        action: int | Array,
        params: EnvParams | None = None,
    ) -> tuple[Array, TimeAwareObservationState, Array, Array, dict[str, Any]]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        time = jnp.where(done, 0, state.time + 1)
        return (
            self._append_time(obs, time),
            TimeAwareObservationState(env_state, time),
            reward,
            done,
            info,
        )
