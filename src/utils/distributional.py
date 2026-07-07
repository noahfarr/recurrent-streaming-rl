import jax
import jax.numpy as jnp


def categorical_projection(tz, probs, atoms):
    num_atoms = atoms.shape[0]
    delta = (atoms[-1] - atoms[0]) / (num_atoms - 1)
    b = jnp.clip((tz - atoms[0]) / delta, 0.0, num_atoms - 1)
    lower = jnp.clip(jnp.floor(b).astype(jnp.int32), 0, num_atoms - 2)
    upper = lower + 1
    return (
        (probs * (upper - b))[..., None] * jax.nn.one_hot(lower, num_atoms)
        + (probs * (b - lower))[..., None] * jax.nn.one_hot(upper, num_atoms)
    ).sum(axis=-2)
