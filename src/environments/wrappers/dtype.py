from typing import Any

import jax.numpy as jnp
from gymnax.wrappers.purerl import GymnaxWrapper
from streamlet.utils.typing import Array, EnvParams, Key


class DtypeWrapper(GymnaxWrapper):
    def step(
        self,
        key: Key,
        state: Any,
        action: int | Array,
        params: EnvParams | None = None,
    ) -> tuple[Array, Any, Array, bool, dict[str, Any]]:
        obs, env_state, reward, done, info = self._env.step(key, state, action, params)
        return (
            obs,
            env_state,
            jnp.asarray(reward).astype(jnp.float32),
            jnp.asarray(done).astype(jnp.bool_),
            info,
        )
