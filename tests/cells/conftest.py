import jax
import jax.numpy as jnp
import pytest

from src.cells import (
    GRUCell,
    GRUConfig,
    MinGRUCell,
    MinGRUConfig,
    RNN,
    RTUCell,
    RTUConfig,
)


@pytest.fixture
def gru_cell():
    return GRUCell(config=GRUConfig(features=4, hidden_dim=6))


@pytest.fixture
def rtu_cell():
    return RTUCell(config=RTUConfig(features=4, hidden_dim=6))


@pytest.fixture
def min_gru_cell():
    return MinGRUCell(config=MinGRUConfig(features=4, hidden_dim=6))


def bptt_grad(cell, params, xs, key):
    bptt = RNN(cell=cell)

    def loss(params, xs):
        carry = bptt.initialize_carry(key, xs.shape[1:])
        total = 0.0
        for x in xs:
            carry, out = bptt.apply(params, carry, x)
            total = total + jnp.sum(out**2)
        return total

    return jax.grad(loss)(params, xs)


def bptt_forward(cell, params, xs, key):
    bptt = RNN(cell=cell)
    carry = bptt.initialize_carry(key, xs.shape[1:])
    outputs = []
    for x in xs:
        carry, out = bptt.apply(params, carry, x)
        outputs.append(out)
    return jnp.stack(outputs)


def max_abs_diff(tree_a, tree_b):
    diffs = jax.tree.map(lambda a, b: jnp.max(jnp.abs(a - b)), tree_a, tree_b)
    return max(jax.tree.leaves(diffs))


def assert_trees_close(tree_a, tree_b, atol=1e-3, rtol=1e-3):
    close = jax.tree.map(
        lambda a, b: bool(jnp.allclose(a, b, atol=atol, rtol=rtol)), tree_a, tree_b
    )
    assert all(jax.tree.leaves(close)), max_abs_diff(tree_a, tree_b)
