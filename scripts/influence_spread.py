import sys

import jax
import jax.numpy as jnp
import lox
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from jax.flatten_util import ravel_pytree

from src import algorithm

CONFIG_DIR = "/home/farr/recurrent-streaming-rl/config"
NUM_SEEDS = 4
NUM_CHUNKS = 2
STEPS_PER_CHUNK = 20_000

ENVIRONMENTS = {
    "memory_chain": [
        "environment=gymnax/bsuite/memory_chain",
        "environment.env_params.max_steps_in_episode=129",
    ],
    "repeat_previous": ["environment=popgymnax/repeat_previous/easy"],
    "count_recall": ["environment=popgymnax/count_recall/easy"],
    "autoencode": ["environment=popgymnax/autoencode/easy"],
    "minatar_breakout": ["environment=streamlet/minatar/breakout"],
    "halfcheetah_masked": [
        "environment=mujoco/halfcheetah",
        "+environment.kwargs.mode=P",
    ],
}


def influences_of(carry):
    if hasattr(carry, "influence"):
        return [carry.influence]
    if isinstance(carry, (tuple, list)):
        return [x for element in carry for x in influences_of(element)]
    if hasattr(carry, "carry"):
        return influences_of(carry.carry)
    return []


def diagonal_gram(influence, cell):
    if cell == "min_gru":
        return jnp.square(influence)
    if cell in ("rtu", "gtu"):
        return jnp.sum(jnp.square(influence), axis=0)
    raise ValueError(f"unsupported cell {cell}")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "repeat_previous"
    cell = sys.argv[2] if len(sys.argv) > 2 else "min_gru"
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
    print(
        f"{'step':>9} {'params':>8}  quantiles ... | fraction of params above threshold x max"
        f"{'p99':>10} {'max':>10} {'p99/p1':>9} {'max/med':>9}"
    )
    for chunk in range(NUM_CHUNKS):
        state, _ = train(
            jax.random.split(chunk_keys[chunk], NUM_SEEDS), state, STEPS_PER_CHUNK
        )
        parts = influences_of(state.carry)
        assert parts, "no influence found in carry"
        grams = [
            jax.vmap(lambda x: diagonal_gram(x, cell))(p).reshape(NUM_SEEDS, -1)
            for p in parts
        ]
        flat = jnp.concatenate(grams, axis=-1)
        peak = jnp.max(flat, axis=-1, keepdims=True)
        share = {
            f"frac>{t:g}": float(jnp.mean(jnp.mean(flat > t * peak, axis=-1)))
            for t in (1e-2, 1e-4, 1e-6, 1e-8)
        }
        quantiles = {
            f"p{q}": float(jnp.mean(jnp.percentile(flat, q, axis=-1)))
            for q in (10, 25, 50, 75, 90, 99)
        }
        print(
            f"{(chunk + 1) * STEPS_PER_CHUNK:>9} {flat.shape[-1]:>8} "
            + " ".join(f"{k}={v:.1e}" for k, v in quantiles.items())
            + "  |  "
            + " ".join(f"{k}={v:.3f}" for k, v in share.items())
        )


if __name__ == "__main__":
    main()
