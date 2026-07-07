import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from jax.experimental import sparse
from src.utils.typing import Array, Carry, PyTree

from .rnn import reset_carry


def _to_diagonal_bcoo(leaf):
    H = leaf.shape[0]
    indices = jnp.stack([jnp.arange(H), jnp.arange(H)], axis=1)
    return sparse.BCOO((leaf, indices), shape=(H, H) + leaf.shape[1:])


@struct.dataclass
class SnAp1Carry:
    carry: Carry
    influence: PyTree


class SnAp1(nn.Module):
    cell: nn.Module

    @nn.compact
    def __call__(
        self, carry: SnAp1Carry, inputs: Array, done: Array | None = None, **kwargs
    ) -> tuple[SnAp1Carry, Array]:
        if done is None:
            done = jnp.zeros((), dtype=jnp.bool_)
        initial_carry = self.cell.initialize_carry(jax.random.key(0), inputs.shape)
        initial_influence = jax.tree.map(
            lambda leaf: jnp.zeros(leaf.shape[1:]),
            self.cell.initialize_influence(jax.random.key(0), inputs.shape),
        )
        carry = SnAp1Carry(
            carry=reset_carry(done, carry.carry, initial_carry),
            influence=reset_carry(done, carry.influence, initial_influence),
        )

        new_carry, state_jacobian, parameter_jacobian = self.cell.local_jacobian(
            carry.carry, inputs
        )
        diagonal_state_jacobian = jnp.diagonal(state_jacobian)
        diagonal_parameter_jacobian = jax.tree.map(
            lambda leaf: jnp.einsum("ii...->i...", leaf), parameter_jacobian
        )

        def update_influence(influence_leaf, parameter_jacobian_leaf):
            broadcast_dims = (1,) * (influence_leaf.ndim - 1)
            rotated = diagonal_state_jacobian.reshape(-1, *broadcast_dims) * influence_leaf
            return rotated + parameter_jacobian_leaf

        next_influence = jax.tree.map(
            update_influence, carry.influence, diagonal_parameter_jacobian
        )

        new_carry = self.cell.inject_influence(
            new_carry, jax.tree.map(_to_diagonal_bcoo, next_influence)
        )

        return SnAp1Carry(carry=new_carry, influence=next_influence), self.cell.output(
            new_carry
        )

    def initialize_carry(self, key, input_shape):
        carry = self.cell.initialize_carry(key, input_shape)
        influence = jax.tree.map(
            lambda leaf: jnp.zeros(leaf.shape[1:]),
            self.cell.initialize_influence(key, input_shape),
        )
        return SnAp1Carry(carry=carry, influence=influence)
