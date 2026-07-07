from typing import Any

import jax.numpy as jnp
from gymnax.wrappers.purerl import GymnaxWrapper
from streamlet.utils.typing import Array, EnvParams, Key


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
    """Zeros out half of a [position, velocity]-concatenated observation.

    Reproduces the masked-MuJoCo POMDP of Ni et al. (2022): mode="P" keeps
    positions observable (velocities masked), mode="V" keeps velocities
    observable (positions masked) -- the letter names what stays visible, not
    what's masked. Assumes the wrapped env's observation is `concat([position,
    velocity])`, as in brax's Ant/HalfCheetah/Hopper/Walker2d.
    """

    def __init__(self, env, mode):
        super().__init__(env)
        self.mode = mode
        self.velocity_size = env.sys.qd_size()

    def _mask(self, obs):
        if self.mode == "P":
            return obs.at[..., -self.velocity_size :].set(0.0)
        if self.mode == "V":
            return obs.at[..., : -self.velocity_size].set(0.0)
        return obs

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        return self._mask(obs), state

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return self._mask(obs), state, reward, done, info


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
