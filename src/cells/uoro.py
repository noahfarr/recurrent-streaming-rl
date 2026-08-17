import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax import struct
from src.utils.typing import Array, Carry

from .rnn import reset_carry


@struct.dataclass
class UOROCarry:
    carry: Carry
    u: Array
    v: Array
    key: Array


class UORO(nn.Module):
    cell: nn.Module
    eps: float = 1e-8

    @nn.compact
    def __call__(
        self, carry: UOROCarry, inputs: Array, done: Array | None = None, **kwargs
    ) -> tuple[UOROCarry, Array]:
        if done is None:
            done = jnp.zeros((), dtype=jnp.bool_)
        initial_carry = self.cell.initialize_carry(jax.random.key(0), inputs.shape)
        carry = UOROCarry(
            carry=reset_carry(done, carry.carry, initial_carry),
            u=reset_carry(done, carry.u, jnp.zeros_like(carry.u)),
            v=reset_carry(done, carry.v, jnp.zeros_like(carry.v)),
            key=carry.key,
        )

        new_carry, forward_u = self.cell.local_jvp(carry.carry, inputs, carry.u)

        key, noise_key = jax.random.split(carry.key)
        noise = jax.random.rademacher(noise_key, carry.u.shape, dtype=carry.u.dtype)

        immediate_v = self.cell.local_vjp(carry.carry, inputs, noise)

        rho_0 = jnp.sqrt(
            (optax.global_norm(carry.v) + self.eps)
            / (optax.global_norm(forward_u) + self.eps)
        )
        rho_1 = jnp.sqrt(
            (optax.global_norm(immediate_v) + self.eps)
            / (optax.global_norm(noise) + self.eps)
        )

        u = jax.lax.stop_gradient(rho_0 * forward_u + rho_1 * noise)
        v = jax.lax.stop_gradient(carry.v / rho_0 + immediate_v / rho_1)

        new_carry = self.cell.inject_influence_rank1(new_carry, u, v)

        return UOROCarry(carry=new_carry, u=u, v=v, key=key), self.cell.output(
            new_carry
        )

    def initialize_carry(self, key, input_shape):
        carry = self.cell.initialize_carry(key, input_shape)
        influence = self.cell.initialize_influence(key, input_shape)
        u = jnp.zeros_like(carry)
        v = jnp.zeros(influence.shape[-1])
        return UOROCarry(carry=carry, u=u, v=v, key=key)
