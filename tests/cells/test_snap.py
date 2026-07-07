import jax
import jax.numpy as jnp

from src.cells import SnAp1

from .conftest import assert_trees_close, bptt_forward, bptt_grad


def test_snap1_forward_matches_bptt(gru_cell):
    cell = gru_cell
    snap = SnAp1(cell=cell)
    key = jax.random.key(0)
    xs = jax.random.normal(jax.random.key(42), (5, cell.config.features))
    carry0 = snap.initialize_carry(key, xs.shape[1:])
    params = snap.init(key, carry0, xs[0])

    carry = carry0
    outputs = []
    for x in xs:
        carry, out = snap.apply(params, carry, x)
        outputs.append(out)
    out_snap = jnp.stack(outputs)

    out_bptt = bptt_forward(cell, params, xs, key)
    assert jnp.max(jnp.abs(out_snap - out_bptt)) < 1e-4


def test_snap1_gradient_matches_bptt_at_single_step(gru_cell):
    cell = gru_cell
    snap = SnAp1(cell=cell)
    key = jax.random.key(0)
    xs = jax.random.normal(jax.random.key(42), (1, cell.config.features))
    carry0 = snap.initialize_carry(key, xs.shape[1:])
    params = snap.init(key, carry0, xs[0])

    def loss(params, xs):
        carry, out = snap.apply(params, carry0, xs[0])
        return jnp.sum(out**2)

    g_snap = jax.grad(loss)(params, xs)
    g_bptt = bptt_grad(cell, params, xs, key)
    assert_trees_close(g_snap, g_bptt)


def _diagonal_snap1_reference_grad(cell, theta, xs):
    def step(h, theta, x):
        r = jax.nn.sigmoid(theta["W_ir"] @ x + theta["W_hr"] @ h + theta["b_r"])
        z = jax.nn.sigmoid(theta["W_iz"] @ x + theta["W_hz"] @ h + theta["b_z"])
        n = jnp.tanh(theta["W_in"] @ x + r * (theta["W_hn"] @ h + theta["b_n"]))
        return (1.0 - z) * n + z * h

    def diagonal_of(full_jacobian):
        return jnp.moveaxis(jnp.diagonal(full_jacobian, axis1=0, axis2=1), -1, 0)

    h = jnp.zeros(cell.config.hidden_dim)
    influence = jax.tree.map(jnp.zeros_like, theta)
    grad = jax.tree.map(jnp.zeros_like, theta)
    for x in xs:
        state_jacobian = jax.jacfwd(step, argnums=0)(h, theta, x)
        parameter_jacobian = jax.jacfwd(step, argnums=1)(h, theta, x)
        new_h = step(h, theta, x)
        diagonal_state_jacobian = jnp.diagonal(state_jacobian)

        influence = jax.tree.map(
            lambda i, p: diagonal_of(p)
            + diagonal_state_jacobian.reshape(-1, *([1] * (i.ndim - 1))) * i,
            influence,
            parameter_jacobian,
        )
        tangent = 2.0 * new_h
        grad = jax.tree.map(
            lambda g, i: g + tangent.reshape(-1, *([1] * (i.ndim - 1))) * i,
            grad,
            influence,
        )
        h = new_h
    return grad


def test_snap1_gradient_matches_independent_reference(gru_cell):
    cell = gru_cell
    snap = SnAp1(cell=cell)
    key = jax.random.key(0)
    xs = jax.random.normal(jax.random.key(42), (6, cell.config.features))
    carry0 = snap.initialize_carry(key, xs.shape[1:])
    params = snap.init(key, carry0, xs[0])
    theta = params["params"]["cell"]

    def loss(params, xs):
        carry = snap.initialize_carry(key, xs.shape[1:])
        total = 0.0
        for x in xs:
            carry, out = snap.apply(params, carry, x)
            total = total + jnp.sum(out**2)
        return total

    g_snap = jax.grad(loss)(params, xs)["params"]["cell"]
    g_reference = _diagonal_snap1_reference_grad(cell, theta, xs)
    assert_trees_close(g_snap, g_reference)
