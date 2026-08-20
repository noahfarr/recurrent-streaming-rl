import jax
import jax.numpy as jnp
import pytest

from src.cells import RTRL, BufferedRTRL, replay_influence, staleness_statistics

from .conftest import assert_trees_close, bptt_grad

CELL_FIXTURES = ["gru_cell", "rtu_cell", "min_gru_cell", "gtu_cell"]


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
            carry, out = rtrl.apply(params, jax.lax.stop_gradient(carry), x)
            total = total + jnp.sum(out**2)
        return total

    g_rtrl = jax.grad(loss)(params, xs)
    g_bptt = bptt_grad(cell, params, xs, key)
    assert_trees_close(g_rtrl, g_bptt)


@pytest.mark.parametrize("cell_fixture", CELL_FIXTURES)
def test_buffered_replay_matches_online_influence_under_frozen_params(
    cell_fixture, request
):
    cell = request.getfixturevalue(cell_fixture)
    buffered = BufferedRTRL(cell=cell, buffer_size=16)
    key = jax.random.key(0)
    xs = jax.random.normal(jax.random.key(42), (7, cell.config.features))
    carry = buffered.initialize_carry(key, xs.shape[1:])
    params = buffered.init(key, carry, xs[0])

    for x in xs:
        carry, _ = buffered.apply(params, carry, x)

    replayed = replay_influence(
        cell, params["params"]["cell"], carry.buffer, carry.length
    )
    assert_trees_close(carry.carry.influence, replayed)
    stats = staleness_statistics(carry.carry.influence, replayed)
    assert float(stats["relative_l2"]) < 1e-6
