import sys

import jax
import jax.numpy as jnp
import lox
from hydra import compose, initialize_config_dir
from jax.flatten_util import ravel_pytree
from hydra.core.hydra_config import HydraConfig

from src import algorithm
from src.cells import replay, staleness_statistics

CONFIG_DIR = "/home/farr/recurrent-streaming-rl/config"
NUM_SEEDS = 4
NUM_CHUNKS = 8
STEPS_PER_CHUNK = 25_000

ENVIRONMENTS = {
    "memory_chain": [
        "environment=gymnax/bsuite/memory_chain",
        "environment.env_params.max_steps_in_episode=129",
    ],
    "repeat_previous": ["environment=popgymnax/repeat_previous/easy"],
    "count_recall": ["environment=popgymnax/count_recall/easy"],
    "autoencode": ["environment=popgymnax/autoencode/easy"],
}


def carry_state(carry):
    while hasattr(carry, "carry"):
        carry = carry.carry
    return carry


def find_cell_params(params):
    found = []

    def search(tree, path):
        if not hasattr(tree, "items"):
            return
        for key, value in tree.items():
            if key == "cell" and hasattr(value, "items") and "cell" in value:
                found.append((path + ["cell", "cell"], value["cell"]))
            search(value, path + [key])

    search(params["params"], [])
    return found


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "memory_chain"
    cell = sys.argv[2] if len(sys.argv) > 2 else "min_gru"
    print(f"environment={name} cell={cell}")

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=stream_ac",
                *ENVIRONMENTS[name],
                f"cell={cell}",
                "mode=rtrl_buffered",
                f"num_seeds={NUM_SEEDS}",
                "total_timesteps=1",
                "num_epochs=1",
            ],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
        agent = algorithm.make(cfg)

    inner_network = agent.network.network
    raw_cell = inner_network.cell.cell

    env, env_params = agent.env, agent.env_params
    action_space = env.action_space(env_params)
    dummy = (
        inner_network.initialize_carry(jax.random.key(0), ()),
        jnp.zeros(env.observation_space(env_params).shape),
        jnp.zeros(getattr(action_space, "shape", ()) or ()),
        jnp.zeros(()),
        jnp.zeros((), bool),
    )
    shapes = jax.eval_shape(lambda k: inner_network.init(k, *dummy), jax.random.key(0))
    template = jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), shapes)
    _, unravel = ravel_pytree(template)

    init = jax.jit(jax.vmap(agent.init))
    train = jax.jit(
        jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None)), static_argnums=(2,)
    )

    key = jax.random.key(0)
    init_key, train_key = jax.random.split(key)
    state = init(jax.random.split(init_key, NUM_SEEDS))

    def measure(cell_params, buffer, length):
        carry, influence = replay(raw_cell, cell_params, buffer, length)
        return carry, influence

    batched_measure = jax.jit(jax.vmap(measure))

    def flat_carry(carry):
        return jnp.concatenate(
            [leaf.reshape(-1) for leaf in jax.tree.leaves(carry_state(carry))]
        )

    def step_change(cell_params, buffer, length):
        prefixes = jnp.arange(1, buffer.shape[0] + 1)
        carries = jax.vmap(
            lambda prefix: flat_carry(replay(raw_cell, cell_params, buffer, prefix)[0])
        )(prefixes)
        active = prefixes[1:] <= length
        deltas = jnp.linalg.norm(carries[1:] - carries[:-1], axis=-1)
        scale = jnp.linalg.norm(carries[1:], axis=-1) + 1e-12
        return jnp.sum(jnp.where(active, deltas / scale, 0.0)) / jnp.maximum(
            jnp.sum(active), 1
        )

    batched_step_change = jax.jit(jax.vmap(step_change))

    chunk_keys = jax.random.split(train_key, NUM_CHUNKS)
    previous_cell_params = None
    print(
        f"{'step':>10} {'len':>5} {'carry rel_l2':>14} {'carry cos':>11} "
        f"{'infl rel_l2':>13} {'infl cos':>10} {'param drift':>12} "
        f"{'stale-25k':>10} {'carry move':>11}"
    )
    for chunk in range(NUM_CHUNKS):
        state, _ = train(
            jax.random.split(chunk_keys[chunk], NUM_SEEDS), state, STEPS_PER_CHUNK
        )

        buffered = state.carry
        length = buffered.length
        assert bool(jnp.all(length <= buffered.buffer.shape[1])), (
            "episode longer than replay buffer; replay would be truncated"
        )

        unraveled = jax.vmap(unravel)(state.params["params"]["raveled"])
        matches = find_cell_params(unraveled)
        assert matches, "could not locate cell parameters"
        _, cell_params = matches[0]

        replayed_carry, replayed_influence = batched_measure(
            cell_params, buffered.buffer, length
        )

        online_carry = carry_state(buffered)
        online_influence = buffered.carry.influence

        carry_stats = staleness_statistics(
            jnp.concatenate(
                [leaf.reshape(NUM_SEEDS, -1) for leaf in jax.tree.leaves(online_carry)],
                axis=-1,
            ),
            jnp.concatenate(
                [
                    leaf.reshape(NUM_SEEDS, -1)
                    for leaf in jax.tree.leaves(carry_state(replayed_carry))
                ],
                axis=-1,
            ),
        )
        influence_stats = staleness_statistics(online_influence, replayed_influence)

        drift, lagged = float("nan"), float("nan")
        if previous_cell_params is not None:
            current_flat = jnp.concatenate(
                [leaf.reshape(-1) for leaf in jax.tree.leaves(cell_params)]
            )
            previous_flat = jnp.concatenate(
                [leaf.reshape(-1) for leaf in jax.tree.leaves(previous_cell_params)]
            )
            drift = float(
                jnp.linalg.norm(current_flat - previous_flat)
                / jnp.linalg.norm(current_flat)
            )
            lagged_carry, _ = batched_measure(
                previous_cell_params, buffered.buffer, length
            )
            lagged = float(
                staleness_statistics(
                    jnp.concatenate(
                        [
                            leaf.reshape(NUM_SEEDS, -1)
                            for leaf in jax.tree.leaves(carry_state(lagged_carry))
                        ],
                        axis=-1,
                    ),
                    jnp.concatenate(
                        [
                            leaf.reshape(NUM_SEEDS, -1)
                            for leaf in jax.tree.leaves(carry_state(replayed_carry))
                        ],
                        axis=-1,
                    ),
                )["relative_l2"]
            )
        previous_cell_params = cell_params
        movement = batched_step_change(cell_params, buffered.buffer, length)

        print(
            f"{(chunk + 1) * STEPS_PER_CHUNK:>10} {int(length[0]):>5} "
            f"{float(carry_stats['relative_l2']):>14.4f} "
            f"{float(carry_stats['cosine_similarity']):>11.4f} "
            f"{float(influence_stats['relative_l2']):>13.4f} "
            f"{float(influence_stats['cosine_similarity']):>10.4f} "
            f"{drift:>12.4f} {lagged:>10.4f} {float(movement.mean()):>11.4f}"
        )


if __name__ == "__main__":
    main()
