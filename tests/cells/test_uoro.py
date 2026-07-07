import jax
import jax.numpy as jnp

from src.cells import UORO

from .conftest import bptt_forward, bptt_grad


def test_uoro_forward_matches_bptt(gru_cell):
    cell = gru_cell
    uoro = UORO(cell=cell)
    key = jax.random.key(0)
    xs = jax.random.normal(jax.random.key(42), (5, cell.config.features))
    carry0 = uoro.initialize_carry(key, xs.shape[1:])
    params = uoro.init(key, carry0, xs[0])

    carry = carry0
    outputs = []
    for x in xs:
        carry, out = uoro.apply(params, carry, x)
        outputs.append(out)
    out_uoro = jnp.stack(outputs)

    out_bptt = bptt_forward(cell, params, xs, key)
    assert jnp.max(jnp.abs(out_uoro - out_bptt)) < 1e-4


def test_uoro_gradient_is_unbiased(gru_cell):
    cell = gru_cell
    uoro = UORO(cell=cell)
    xs = jax.random.normal(jax.random.key(1), (2, cell.config.features))
    base_key = jax.random.key(0)
    carry0 = uoro.initialize_carry(base_key, xs.shape[1:])
    params = uoro.init(base_key, carry0, xs[0])

    def loss(params, xs, key):
        carry = uoro.initialize_carry(key, xs.shape[1:])
        total = 0.0
        for x in xs:
            carry, out = uoro.apply(params, carry, x)
            total = total + jnp.sum(out**2)
        return total

    num_samples = 4000
    keys = jax.random.split(jax.random.key(2), num_samples)
    grad_fn = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, None, 0)))
    g_samples = grad_fn(params, xs, keys)
    g_mean = jax.tree.map(lambda a: jnp.mean(a, axis=0), g_samples)

    g_true = bptt_grad(cell, params, xs, base_key)

    diff_norm = jnp.sqrt(
        sum(
            jnp.sum((a - b) ** 2)
            for a, b in zip(jax.tree.leaves(g_mean), jax.tree.leaves(g_true))
        )
    )
    true_norm = jnp.sqrt(sum(jnp.sum(b**2) for b in jax.tree.leaves(g_true)))
    assert diff_norm / true_norm < 0.15
