from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def inject(subscripts, carry, parameters, influence):
    return carry


@inject.defjvp
def inject_jvp(subscripts, primals, tangents):
    carry, _, influence = primals
    carry_tangent, parameter_tangent, _ = tangents
    return carry, carry_tangent + jnp.einsum(subscripts, influence, parameter_tangent)


@jax.custom_jvp
def inject_diagonal(carry, parameters, influence, unit_index):
    return carry


@inject_diagonal.defjvp
def inject_diagonal_jvp(primals, tangents):
    carry, _, influence, unit_index = primals
    carry_tangent, parameter_tangent, _, _ = tangents
    contribution = jax.ops.segment_sum(
        influence * parameter_tangent, unit_index, num_segments=carry.shape[0]
    )
    return carry, carry_tangent + contribution


@jax.custom_jvp
def inject_rank1(carry, parameters, u, v):
    return carry


@inject_rank1.defjvp
def inject_rank1_jvp(primals, tangents):
    carry, _, u, v = primals
    carry_tangent, parameter_tangent, _, _ = tangents
    return carry, carry_tangent + u * (v @ parameter_tangent)
