import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from src.utils.typing import Array, Carry

from .rnn import reset_carry


@struct.dataclass
class RTRLCarry:
    carry: Carry
    influence: Array


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

        next_influence = (
            self.cell.propagate_influence(state_jacobian, carry.influence)
            + parameter_jacobian
        )

        new_carry = self.cell.inject_influence(new_carry, next_influence)

        return RTRLCarry(carry=new_carry, influence=next_influence), self.cell.output(
            new_carry
        )

    def initialize_carry(self, key, input_shape):
        carry = self.cell.initialize_carry(key, input_shape)
        influence = self.cell.initialize_influence(key, input_shape)
        return RTRLCarry(carry=carry, influence=influence)


@struct.dataclass
class BufferedRTRLCarry:
    carry: RTRLCarry
    buffer: Array
    length: Array


class BufferedRTRL(RTRL):
    buffer_size: int = 256

    @nn.compact
    def __call__(
        self,
        carry: BufferedRTRLCarry,
        inputs: Array,
        done: Array | None = None,
        **kwargs,
    ) -> tuple[BufferedRTRLCarry, Array]:
        if done is None:
            done = jnp.zeros((), dtype=jnp.bool_)
        buffer = jnp.where(done, jnp.zeros_like(carry.buffer), carry.buffer)
        length = jnp.where(done, 0, carry.length)
        buffer = buffer.at[length % self.buffer_size].set(inputs)

        inner_carry, output = super().__call__(carry.carry, inputs, done, **kwargs)

        return (
            BufferedRTRLCarry(carry=inner_carry, buffer=buffer, length=length + 1),
            output,
        )

    def initialize_carry(self, key, input_shape):
        return BufferedRTRLCarry(
            carry=super().initialize_carry(key, input_shape),
            buffer=jnp.zeros((self.buffer_size, self.cell.config.features)),
            length=jnp.zeros((), dtype=jnp.int32),
        )


def replay_influence(cell, cell_params, buffer, length):
    initial_carry = cell.initialize_carry(jax.random.key(0), buffer.shape[-1:])
    initial_influence = cell.initialize_influence(jax.random.key(0), buffer.shape[-1:])

    def step(scan_carry, indexed):
        index, inputs = indexed
        carry, influence = scan_carry
        new_carry, state_jacobian, parameter_jacobian = cell.apply(
            {"params": cell_params}, carry, inputs, method=cell.local_jacobian
        )
        new_influence = (
            cell.apply(
                {"params": cell_params},
                state_jacobian,
                influence,
                method=cell.propagate_influence,
            )
            + parameter_jacobian
        )
        active = index < length
        carry = jax.tree.map(
            lambda new, old: jnp.where(active, new, old), new_carry, carry
        )
        influence = jnp.where(active, new_influence, influence)
        return (carry, influence), None

    (_, influence), _ = jax.lax.scan(
        step,
        (initial_carry, initial_influence),
        (jnp.arange(buffer.shape[0]), buffer),
    )
    return influence


def staleness_statistics(online, replay):
    online = online.reshape(-1)
    replay = replay.reshape(-1)
    difference_norm = jnp.linalg.norm(online - replay)
    replay_norm = jnp.linalg.norm(replay)
    online_norm = jnp.linalg.norm(online)
    return {
        "relative_l2": difference_norm / (replay_norm + 1e-12),
        "cosine_similarity": jnp.dot(online, replay)
        / (online_norm * replay_norm + 1e-12),
        "online_norm": online_norm,
        "replay_norm": replay_norm,
    }
