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
        out, variables = self.network.apply(
            unravel(params), carry, *args, mutable=["auxiliary_losses"], **kwargs
        )
        self.sow("auxiliary_losses", "inner", variables.get("auxiliary_losses", {}))
        return out

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple = ()) -> Carry:
        return self.network.initialize_carry(key, input_shape)
