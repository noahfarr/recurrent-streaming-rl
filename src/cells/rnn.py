import jax
import jax.numpy as jnp
from flax import linen as nn

from src.utils.typing import Array, Carry, Key


def broadcast_done(done: Array, leaf: Array) -> Array:
    while done.ndim != leaf.ndim:
        done = done[..., None]
    return done


def reset_carry(done: Array, carry: Carry, initial_carry: Carry) -> Carry:
    return jax.tree.map(
        lambda initial, leaf: jnp.where(broadcast_done(done, leaf), initial, leaf),
        initial_carry,
        carry,
    )


class RNN(nn.Module):
    cell: nn.Module

    @nn.compact
    def __call__(
        self, carry: Carry, inputs: Array, done: Array | None = None, **kwargs
    ) -> tuple[Carry, Array]:
        single_step = inputs.ndim == 1
        if single_step:
            inputs = inputs[None]
            done = done if done is None else jnp.asarray(done)[None]
        if done is None:
            done = jnp.zeros(inputs.shape[0], dtype=jnp.bool_)
        initial_carry = self.cell.initialize_carry(jax.random.key(0), inputs.shape[1:])

        def scan_fn(cell, carry, inputs):
            features, done = inputs
            carry = reset_carry(done, carry, initial_carry)
            return cell(carry, features)

        scan = nn.scan(
            scan_fn,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )
        carry, outputs = scan(self.cell, carry, (inputs, done))
        if single_step:
            outputs = outputs[0]
        return carry, outputs

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple = ()) -> Carry:
        return self.cell.initialize_carry(key, input_shape)
