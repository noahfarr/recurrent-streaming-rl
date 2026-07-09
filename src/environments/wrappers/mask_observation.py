import jax
import jax.numpy as jnp
import lox
from gymnax.wrappers.purerl import GymnaxWrapper
from streamlet.utils.typing import PyTree


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
