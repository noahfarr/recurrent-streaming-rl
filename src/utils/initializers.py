import jax
import jax.numpy as jnp


def constant_init(key, shape, value=0.0):
    return jnp.full(shape, value)


def forget_bias_init(key, shape, f_min=0.0, f_max=1.0):
    u = jax.random.uniform(key, shape=shape)
    f = u * (f_max - f_min) + f_min
    f = jnp.clip(f, 1e-7, 1.0 - 1e-7)
    return jnp.log(-jnp.log(f))
