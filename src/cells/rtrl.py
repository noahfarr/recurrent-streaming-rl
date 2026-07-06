import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from src.utils.typing import Array, Carry, PyTree


@struct.dataclass
class RTRLCarry:
    carry: Carry
    influence: PyTree


class RTRL(nn.Module):
    cell: nn.Module

    @nn.compact
    def __call__(
        self, carry: RTRLCarry, inputs: Array, **kwargs
    ) -> tuple[RTRLCarry, Array]:
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
