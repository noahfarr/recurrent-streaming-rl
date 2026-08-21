import sys

import jax
import jax.numpy as jnp
import lox
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from jax.flatten_util import ravel_pytree

from src import algorithm

CONFIG_DIR = "/home/farr/recurrent-streaming-rl/config"
NUM_SEEDS = 2
NUM_CHUNKS = 5
STEPS_PER_CHUNK = 20_000

ENVIRONMENTS = {
    "repeat_previous": ["environment=popgymnax/repeat_previous/easy"],
    "memory_chain": ["environment=gymnax/bsuite/memory_chain"],
    "relearning": ["environment=foragax/never_ending_relearning"],
}


def influences_of(carry):
    if hasattr(carry, "influence"):
        return [carry.influence]
    if isinstance(carry, (tuple, list)):
        return [x for element in carry for x in influences_of(element)]
    if hasattr(carry, "carry"):
        return influences_of(carry.carry)
    return []


def spectrum_report(influence):
    matrix = influence.reshape(influence.shape[0], -1)
    singular = jnp.linalg.svd(matrix, compute_uv=False)
    total = jnp.sum(jnp.square(singular))
    cumulative = jnp.cumsum(jnp.square(singular)) / (total + 1e-30)
    stable_rank = total / (jnp.square(singular[0]) + 1e-30)
    return singular, cumulative, stable_rank


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "repeat_previous"
    cell = sys.argv[2] if len(sys.argv) > 2 else "gru"
    print(f"environment={name} cell={cell} seeds={NUM_SEEDS}")

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=stream_ac",
                *ENVIRONMENTS[name],
                f"cell={cell}",
                "mode=rtrl",
                f"num_seeds={NUM_SEEDS}",
                "total_timesteps=1",
                "num_epochs=1",
            ],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
        agent = algorithm.make(cfg)

    init = jax.jit(jax.vmap(agent.init))
    train = jax.jit(
        jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None)), static_argnums=(2,)
    )

    key = jax.random.key(0)
    init_key, train_key = jax.random.split(key)
    state = init(jax.random.split(init_key, NUM_SEEDS))

    chunk_keys = jax.random.split(train_key, NUM_CHUNKS)
    for chunk in range(NUM_CHUNKS):
        state, _ = train(
            jax.random.split(chunk_keys[chunk], NUM_SEEDS), state, STEPS_PER_CHUNK
        )
        influence = influences_of(state.carry)[0][0]
        singular, cumulative, stable_rank = spectrum_report(influence)
        n = singular.shape[0]
        marks = [1, 2, 4, 8, 16, 32]
        marks = [m for m in marks if m <= n]
        captured = " ".join(
            f"r{m}={float(cumulative[m - 1]):.3f}" for m in marks
        )
        print(
            f"step={(chunk + 1) * STEPS_PER_CHUNK:>8} shape={tuple(influence.reshape(influence.shape[0], -1).shape)} "
            f"stable_rank={float(stable_rank):5.2f}  {captured}  "
            f"s1/s2={float(singular[0] / (singular[1] + 1e-30)):6.2f}"
        )


if __name__ == "__main__":
    main()
