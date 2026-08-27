import jax.numpy as jnp
import numpy as np
from streamlet.optimizers import Autostep, AutostepConfig


def make(**kwargs):
    return Autostep(cfg=AutostepConfig(**kwargs))


def test_effective_step_size_never_exceeds_one():
    optimizer = make(alpha_init=10.0, mu=1e-2)
    params = {"w": jnp.zeros((4,))}
    state = optimizer.init(params)
    feature = {"w": jnp.array([1.0, -2.0, 0.5, 3.0])}

    for _ in range(50):
        _, state = optimizer.update(state, feature, feature, jnp.float32(1.0))
        effective = float(
            jnp.sum(state.alpha["w"] * jnp.abs(feature["w"] * feature["w"]))
        )
        assert effective <= 1.0 + 1e-4


def test_meta_update_is_invariant_to_target_scale():
    feature = {"w": jnp.ones((2,))}

    def final_alpha(scale):
        optimizer = make(alpha_init=0.1, mu=1e-2)
        state = optimizer.init({"w": jnp.zeros((2,))})
        for _ in range(30):
            _, state = optimizer.update(
                state, feature, feature, jnp.float32(scale)
            )
        return np.asarray(state.alpha["w"])

    np.testing.assert_allclose(final_alpha(1.0), final_alpha(1000.0), rtol=1e-3)


def test_stays_finite_when_gradient_and_trace_anticorrelate():
    optimizer = make(alpha_init=0.1, mu=1e-2)
    state = optimizer.init({"w": jnp.zeros((3,))})
    gradient = {"w": jnp.array([1.0, -1.0, 2.0])}
    trace = {"w": jnp.array([-1.0, 3.0, -0.5])}

    for step in range(200):
        updates, state = optimizer.update(
            state, gradient, trace, jnp.float32((-1.0) ** step)
        )
        assert bool(jnp.isfinite(updates["w"]).all())
    assert bool(jnp.isfinite(state.h["w"]).all())
    assert bool(jnp.isfinite(state.alpha["w"]).all())
