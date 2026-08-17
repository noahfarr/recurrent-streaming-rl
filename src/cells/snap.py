import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from src.utils.typing import Array, Carry

from .rnn import reset_carry


@struct.dataclass
class SnAp1Carry:
    carry: Carry
    influence: Array


class SnAp1(nn.Module):
    cell: nn.Module

    @nn.compact
    def __call__(
        self, carry: SnAp1Carry, inputs: Array, done: Array | None = None, **kwargs
    ) -> tuple[SnAp1Carry, Array]:
        if done is None:
            done = jnp.zeros((), dtype=jnp.bool_)
        initial_carry = self.cell.initialize_carry(jax.random.key(0), inputs.shape)
        initial_influence = jnp.zeros(
            self.cell.initialize_influence(jax.random.key(0), inputs.shape).shape[-1]
        )
        carry = SnAp1Carry(
            carry=reset_carry(done, carry.carry, initial_carry),
            influence=reset_carry(done, carry.influence, initial_influence),
        )

        new_carry, diagonal_state_jacobian, diagonal_parameter_jacobian = (
            self.cell.local_jacobian_diagonal(carry.carry, inputs)
        )

        next_influence = (
            diagonal_state_jacobian[self.cell.unit_index] * carry.influence
            + diagonal_parameter_jacobian
        )

        new_carry = self.cell.inject_influence_diagonal(new_carry, next_influence)

        return SnAp1Carry(carry=new_carry, influence=next_influence), self.cell.output(
            new_carry
        )

    def initialize_carry(self, key, input_shape):
        carry = self.cell.initialize_carry(key, input_shape)
        influence = jnp.zeros(
            self.cell.initialize_influence(key, input_shape).shape[-1]
        )
        return SnAp1Carry(carry=carry, influence=influence)
