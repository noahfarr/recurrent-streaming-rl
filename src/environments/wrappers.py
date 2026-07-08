from typing import Any

import jax
import jax.numpy as jnp
import lox
from gymnax.wrappers.purerl import GymnaxWrapper
from streamlet.utils.typing import Array, EnvParams, Key, PyTree


class ClipActionWrapper(GymnaxWrapper):

    def __init__(self, env, low=None, high=None):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action, params=None):
        action_space = self._env.action_space(params)
        low = action_space.low if self.low is None else self.low
        high = action_space.high if self.high is None else self.high
        action = jnp.clip(action, low, high)
        return self._env.step(key, state, action, params)


class MaskObservationWrapper(GymnaxWrapper):
    def __init__(self, env, mask: PyTree):
        super().__init__(env)
        self.mask = mask
        leaves = jax.tree.leaves(mask)
        total = sum(leaf.size for leaf in leaves)
        masked = sum((leaf == 0).sum() for leaf in leaves)
        self.mask_rate = jnp.asarray(masked / total, dtype=jnp.float32)

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        return jax.tree.map(lambda o, m: o * m, obs, self.mask), state

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        lox.log({"mask_observation/mask_rate": self.mask_rate})
        obs = jax.tree.map(lambda o, m: o * m, obs, self.mask)
        return obs, state, reward, done, info


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
