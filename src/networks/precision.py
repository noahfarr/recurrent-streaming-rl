import jax.numpy as jnp


def compute_dtype(cfg):
    return jnp.dtype(cfg.get("precision", "float32"))
