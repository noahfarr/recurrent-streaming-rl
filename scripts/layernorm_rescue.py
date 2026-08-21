import sys

import jax
import jax.numpy as jnp
import lox
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig

from src import algorithm

CONFIG_DIR = "/home/farr/recurrent-streaming-rl/config"
NUM_SEEDS = 4
NUM_CHUNKS = 12
STEPS_PER_CHUNK = 25_000


def influences_of(carry):
    if hasattr(carry, "influence"):
        return [carry.influence]
    if isinstance(carry, (tuple, list)):
        return [x for element in carry for x in influences_of(element)]
    if hasattr(carry, "carry"):
        return influences_of(carry.carry)
    return []


def main():
    layer_norm = sys.argv[1] if len(sys.argv) > 1 else "true"
    trust = sys.argv[2] if len(sys.argv) > 2 else "0.0"
    cell = sys.argv[3] if len(sys.argv) > 3 else "rtu"
    print(f"layer_norm={layer_norm} trust_region={trust} cell={cell} seeds={NUM_SEEDS}")

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=stream_ac",
                "environment=foragax/never_ending_relearning",
                f"cell={cell}",
                "mode=rtrl",
                f"+layer_norm={layer_norm}",
                f"+mode.wrapper.influence_trust_region={trust}",
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
    print(f"{'step':>9} {'reward':>10} {'|td|':>9} {'value':>10} {'|S|_F':>10}")
    for chunk in range(NUM_CHUNKS):
        state, logs = train(
            jax.random.split(chunk_keys[chunk], NUM_SEEDS), state, STEPS_PER_CHUNK
        )
        parts = influences_of(state.carry)
        norms = [
            jnp.sqrt(jnp.sum(jnp.square(p.reshape(NUM_SEEDS, -1)), axis=-1))
            for p in parts
        ]
        norm = float(jnp.mean(jnp.stack(norms)))

        def metric(name):
            value = logs.get(name)
            return float(jnp.nanmean(value)) if value is not None else float("nan")

        print(
            f"{(chunk + 1) * STEPS_PER_CHUNK:>9} "
            f"{metric('normalize_reward/mean'):>10.4f} "
            f"{metric('critic/absolute_td_error'):>9.4f} "
            f"{metric('critic/value'):>10.4f} "
            f"{norm:>10.3e}"
        )


if __name__ == "__main__":
    main()
