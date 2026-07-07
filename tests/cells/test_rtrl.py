import jax
import jax.numpy as jnp
import pytest

from src.cells import RTRL

from .conftest import assert_trees_close, bptt_grad

CELL_FIXTURES = ["gru_cell", "rtu_cell", "min_gru_cell"]


@pytest.mark.parametrize("T", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("cell_fixture", CELL_FIXTURES)
def test_rtrl_gradient_matches_bptt(cell_fixture, T, request):
    cell = request.getfixturevalue(cell_fixture)
    rtrl = RTRL(cell=cell)
    key = jax.random.key(0)
    xs = jax.random.normal(jax.random.key(42), (T, cell.config.features))
    carry0 = rtrl.initialize_carry(key, xs.shape[1:])
    params = rtrl.init(key, carry0, xs[0])

    def loss(params, xs):
        carry = rtrl.initialize_carry(key, xs.shape[1:])
        total = 0.0
        for x in xs:
            carry, out = rtrl.apply(params, carry, x)
            total = total + jnp.sum(out**2)
        return total

    g_rtrl = jax.grad(loss)(params, xs)
    g_bptt = bptt_grad(cell, params, xs, key)
    assert_trees_close(g_rtrl, g_bptt)
