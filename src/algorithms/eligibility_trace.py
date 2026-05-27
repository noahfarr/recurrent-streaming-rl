import math
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import struct
from memorax.utils import Timestep
from memorax.utils.typing import Array, Carry, PyTree

from src.utils.staleness_buffer import (
    StalenessBuffer,
    StalenessBufferState,
    StalenessStatistics,
)


def trace_horizon_window(gamma: float, trace_lambda: float) -> int:
    gamma_lambda = gamma * trace_lambda
    if gamma_lambda >= 1.0:
        raise ValueError(f"γλ = {gamma_lambda} is not contractive; pick γλ < 1.")
    return math.ceil(5.0 / (1.0 - gamma_lambda))


def broadcast(scalar_batch: Array, target_leaf: Array) -> Array:
    """Broadcasts a batch of scalars (shape (N,)) to match a batched leaf (shape (N, ...))."""
    return scalar_batch[(slice(None),) + (None,) * (target_leaf.ndim - 1)]


@struct.dataclass
class TraceState:
    trace: PyTree
    drift: PyTree | None = None
    correction: PyTree | None = None
    parameters: PyTree | None = None
    buffer: StalenessBufferState | None = None

    @property
    def taylor(self) -> PyTree:
        if self.correction is not None:
            return jax.tree.map(jnp.add, self.trace, self.correction)
        return self.trace


@dataclass
class Trace:
    buffer_capacity: int = 0
    staleness_interval: int = 1
    has_aux: bool = False

    def init(
        self,
        parameters: PyTree,
        num_envs: int,
        *,
        taylor: bool = False,
        timestep: Timestep | None = None,
        carry: Carry | None = None,
        action: Array | None = None,
        aux: Array | None = None,
    ) -> TraceState:
        batched_parameters = jax.tree.map(
            lambda p: jnp.broadcast_to(p, (num_envs, *p.shape)),
            parameters,
        )
        buffer = None
        if self.buffer_capacity > 0:
            if timestep is None or action is None:
                raise ValueError("Buffered trace requires timestep and action samples.")
            sb = StalenessBuffer(capacity=self.buffer_capacity)
            reset = jnp.zeros((num_envs,), dtype=jnp.bool_)
            buffer = sb.init(timestep, action, reset, carry, aux=aux)
        return TraceState(
            trace=jax.tree.map(jnp.zeros_like, batched_parameters),
            drift=jax.tree.map(jnp.zeros_like, batched_parameters) if taylor else None,
            correction=(
                jax.tree.map(jnp.zeros_like, batched_parameters) if taylor else None
            ),
            parameters=batched_parameters if taylor else None,
            buffer=buffer,
        )

    def update(
        self,
        state: TraceState,
        gradient: PyTree,
        discount: Array,
        parameters: PyTree | None = None,
        gradient_function: Callable | None = None,
    ) -> TraceState:
        num_environments, *_ = discount.shape
        next_trace = jax.tree.map(
            lambda trace_leaf, gradient_leaf: broadcast(discount, trace_leaf)
            * trace_leaf
            + gradient_leaf,
            state.trace,
            gradient,
        )

        if state.drift is None:
            return state.replace(trace=next_trace)

        if parameters is None or state.parameters is None or gradient_function is None:
            raise ValueError(
                "Taylor trace update requires current parameters and gradient_function."
            )

        batched_parameters = jax.tree.map(
            lambda parameter: jnp.broadcast_to(
                parameter, (num_environments, *parameter.shape)
            ),
            parameters,
        )
        delta_parameters = jax.tree.map(
            jnp.subtract, batched_parameters, state.parameters
        )

        def per_environment_hvp(index, vector):
            def value_at(p):
                value, _ = gradient_function(p)
                if self.has_aux:
                    value, *_ = value
                return value[index]

            _, hvp = jax.jvp(
                jax.grad(value_at),
                (parameters,),
                (vector,),
            )
            return hvp

        hvp = jax.vmap(per_environment_hvp)(jnp.arange(num_environments), state.drift)

        correction = jax.tree.map(
            lambda correction_leaf, hvp_leaf: broadcast(discount, correction_leaf)
            * correction_leaf
            + hvp_leaf,
            state.correction,
            hvp,
        )
        drift = jax.tree.map(
            lambda drift_leaf, delta_leaf: broadcast(discount, drift_leaf) * drift_leaf
            + delta_leaf,
            state.drift,
            delta_parameters,
        )

        return state.replace(
            trace=next_trace,
            drift=drift,
            correction=correction,
            parameters=batched_parameters,
        )

    def reset(self, state: TraceState, mask: Array) -> TraceState:
        def zero_where(pytree):
            return jax.tree.map(
                lambda x: jnp.where(broadcast(mask, x), jnp.zeros_like(x), x),
                pytree,
            )

        return state.replace(
            trace=zero_where(state.trace),
            drift=zero_where(state.drift) if state.drift is not None else None,
            correction=(
                zero_where(state.correction) if state.correction is not None else None
            ),
        )

    def observe(
        self,
        state: TraceState,
        timestep: Timestep,
        action: Array,
        reset: Array,
        loss_function: Callable,
        parameters: PyTree,
        aux: Array | None = None,
    ) -> TraceState:
        if state.buffer is None or self.buffer_capacity == 0:
            return state
        sb = StalenessBuffer(capacity=self.buffer_capacity)

        def evict(carry, ts, a):
            _, next_carry = loss_function(parameters, ts, carry, a)
            return next_carry

        return state.replace(
            buffer=sb.add(state.buffer, timestep, action, reset, evict, aux=aux)
        )

    def compute_staleness(
        self,
        state: TraceState,
        parameters: PyTree,
        loss_function: Callable,
        gamma_lambda: float,
        step: int,
        *,
        label: str,
    ) -> dict[str, Array]:
        if state.buffer is None:
            return {}

        buffer_state = state.buffer
        num_envs, capacity, *_ = buffer_state.reset.shape
        chronological = (buffer_state.index + jnp.arange(capacity)) % capacity
        discount = jnp.broadcast_to(jnp.float32(gamma_lambda), (num_envs,))

        initial_trace = TraceState(trace=jax.tree.map(jnp.zeros_like, state.trace))
        initial_carry = buffer_state.initial_carry

        aux_buffer = buffer_state.aux

        def step_fn(scan_state, idx):
            trace_state, carry = scan_state
            timestep = jax.tree.map(lambda leaf: leaf[:, idx], buffer_state.timestep)
            action = buffer_state.action[:, idx]
            reset = buffer_state.reset[:, idx]
            aux_slice = aux_buffer[:, idx] if aux_buffer is not None else None

            def loss(p):
                if aux_slice is None:
                    value, next_carry = loss_function(p, timestep, carry, action)
                else:
                    value, next_carry = loss_function(
                        p, timestep, carry, action, aux_slice
                    )
                if self.has_aux:
                    value, *_ = value
                return value, next_carry

            value, vjp_fn, next_carry = jax.vjp(loss, parameters, has_aux=True)
            batch, *_ = value.shape
            (gradient,) = jax.vmap(vjp_fn)(jnp.eye(batch, dtype=value.dtype))
            trace_state = self.update(trace_state, gradient, discount)
            trace_state = self.reset(trace_state, reset)
            return (trace_state, next_carry), None

        def do_compute(_):
            (trace_state, _), _ = jax.lax.scan(
                step_fn, (initial_trace, initial_carry), chronological
            )
            is_full = buffer_state.filled >= capacity
            online, replay = state.taylor, trace_state.trace
            results = StalenessStatistics.compute(online, replay).to_dict(label)
            return jax.tree.map(lambda x: jnp.where(is_full, x, 0.0), results)

        zero_stats = StalenessStatistics(
            l_1=jnp.float32(0.0),
            max_relative_drift=jnp.float32(0.0),
            l_2=jnp.float32(0.0),
            l_inf=jnp.float32(0.0),
            l1_top_1=jnp.float32(0.0),
            l1_top_5=jnp.float32(0.0),
            l1_top_10=jnp.float32(0.0),
            l1_top_25=jnp.float32(0.0),
            l1_top_50=jnp.float32(0.0),
            cosine_similarity=jnp.float32(0.0),
            cosine_top_1=jnp.float32(0.0),
            cosine_top_5=jnp.float32(0.0),
            cosine_top_10=jnp.float32(0.0),
            cosine_top_25=jnp.float32(0.0),
            cosine_top_50=jnp.float32(0.0),
            participation=jnp.float32(0.0),
        )
        zeros = zero_stats.to_dict(label)

        if self.staleness_interval <= 1:
            return do_compute(None)
        return jax.lax.cond(
            step % self.staleness_interval == 0,
            do_compute,
            lambda _: zeros,
            None,
        )
