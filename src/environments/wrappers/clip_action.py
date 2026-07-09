import jax.numpy as jnp
from gymnax.wrappers.purerl import GymnaxWrapper


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
