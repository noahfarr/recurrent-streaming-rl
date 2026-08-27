import jax.numpy as jnp
import numpy as np
from streamlet.optimizers import TIDBD, TIDBDConfig


def make(**kwargs):
    return TIDBD(cfg=TIDBDConfig(**kwargs))


def test_zero_meta_step_size_reduces_to_fixed_step_td():
    optimizer = make(alpha_init=0.05, theta=0.0)
    params = {"w": jnp.zeros((3,))}
    state = optimizer.init(params)
    gradient = {"w": jnp.array([1.0, -2.0, 0.5])}
    trace = {"w": jnp.array([1.0, -2.0, 0.5])}

    for _ in range(5):
        updates, state = optimizer.update(state, gradient, trace, jnp.float32(0.3))
        np.testing.assert_allclose(
            updates["w"], 0.05 * 0.3 * trace["w"], rtol=1e-5
        )


def test_step_size_grows_under_persistently_correlated_updates():
    optimizer = make(alpha_init=0.01, theta=0.1)
    params = {"w": jnp.zeros((1,))}
    state = optimizer.init(params)
    feature = {"w": jnp.ones((1,))}

    initial = float(jnp.exp(state.beta["w"][0]))
    for _ in range(20):
        _, state = optimizer.update(state, feature, feature, jnp.float32(1.0))
    assert float(jnp.exp(state.beta["w"][0])) > initial


def test_step_size_shrinks_under_alternating_updates():
    optimizer = make(alpha_init=0.1, theta=0.1)
    params = {"w": jnp.zeros((1,))}
    state = optimizer.init(params)
    feature = {"w": jnp.ones((1,))}

    initial = float(jnp.exp(state.beta["w"][0]))
    sign = 1.0
    for _ in range(20):
        _, state = optimizer.update(state, feature, feature, jnp.float32(sign))
        sign = -sign
    assert float(jnp.exp(state.beta["w"][0])) < initial


def test_adapt_paths_freezes_step_sizes_outside_the_named_layer():
    optimizer = make(alpha_init=0.02, theta=0.1, adapt_paths=("head",))
    params = {"head": jnp.zeros((1,)), "torso": jnp.zeros((1,))}
    state = optimizer.init(params)
    feature = {"head": jnp.ones((1,)), "torso": jnp.ones((1,))}

    for _ in range(10):
        _, state = optimizer.update(state, feature, feature, jnp.float32(1.0))

    assert float(jnp.exp(state.beta["head"][0])) > 0.02
    np.testing.assert_allclose(float(jnp.exp(state.beta["torso"][0])), 0.02, rtol=1e-5)
