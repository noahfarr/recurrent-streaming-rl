import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from src.utils.typing import Array, Carry, PyTree

from .rnn import reset_carry


@struct.dataclass
class RTRLCarry:
    carry: Carry
    influence: PyTree


class RTRL(nn.Module):
    cell: nn.Module

    @nn.compact
    def __call__(
        self, carry: RTRLCarry, inputs: Array, done: Array | None = None, **kwargs
    ) -> tuple[RTRLCarry, Array]:
        if done is None:
            done = jnp.zeros((), dtype=jnp.bool_)
        initial_carry = self.cell.initialize_carry(jax.random.key(0), inputs.shape)
        initial_influence = self.cell.initialize_influence(
            jax.random.key(0), inputs.shape
        )
        carry = RTRLCarry(
            carry=reset_carry(done, carry.carry, initial_carry),
            influence=reset_carry(done, carry.influence, initial_influence),
        )

        new_carry, state_jacobian, parameter_jacobian = self.cell.local_jacobian(
            carry.carry, inputs
        )

        def update_influence(influence_unit, parameter_jacobian_unit):
            rotated = jax.vmap(
                lambda state_jacobian, influence: state_jacobian @ influence,
                in_axes=(0, 1),
                out_axes=1,
            )(state_jacobian, influence_unit)
            return rotated + parameter_jacobian_unit

        next_influence = jax.tree.map(
            update_influence, carry.influence, parameter_jacobian
        )

        new_carry = self.cell.inject_influence(new_carry, next_influence)

        return RTRLCarry(carry=new_carry, influence=next_influence), self.cell.output(
            new_carry
        )

    def initialize_carry(self, key, input_shape):
        carry = self.cell.initialize_carry(key, input_shape)
        influence = self.cell.initialize_influence(key, input_shape)
        return RTRLCarry(carry=carry, influence=influence)
