import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from src.utils.typing import Carry, Key


class Ravel(nn.Module):
    network: nn.Module

    @nn.compact
    def __call__(self, carry: Carry, *args, **kwargs):
        shapes = jax.eval_shape(
            lambda key: self.network.init(key, carry, *args), jax.random.key(0)
        )
        template = jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), shapes)
        _, unravel = ravel_pytree(template)

        params = self.param(
            "raveled",
            lambda key: ravel_pytree(self.network.init(key, carry, *args))[0],
        )
        return self.network.apply(unravel(params), carry, *args, **kwargs)

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple = ()) -> Carry:
        return self.network.initialize_carry(key, input_shape)
