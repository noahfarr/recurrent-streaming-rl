import sys

import jax
import jax.numpy as jnp
import lox
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig

from src import algorithm

CONFIG_DIR = "/home/farr/recurrent-streaming-rl/config"
NUM_SEEDS = 4
NUM_CHUNKS = 32
STEPS_PER_CHUNK = 25_000

ENVIRONMENTS = {
    "relearning": ["environment=foragax/never_ending_relearning"],
    "unending": ["environment=foragax/unending_tasks"],
}


def influences_of(carry):
    if hasattr(carry, "influence"):
        return [carry.influence]
    if isinstance(carry, (tuple, list)):
        return [x for element in carry for x in influences_of(element)]
    if hasattr(carry, "carry"):
        return influences_of(carry.carry)
    return []


def parameter_gram(influence, cell):
    if cell == "min_gru":
        return jnp.square(influence)
    return jnp.sum(jnp.square(influence), axis=0)


def unit_gram(influence, cell):
    if cell == "min_gru":
        return jnp.sum(jnp.square(influence), axis=-1)
    return jnp.sum(jnp.square(influence), axis=(0, -1))


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "relearning"
    cell = sys.argv[2] if len(sys.argv) > 2 else "rtu"
    power = sys.argv[3] if len(sys.argv) > 3 else "0.0"
    print(f"environment={name} cell={cell} power={power} seeds={NUM_SEEDS} period=100000")

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=stream_ac",
                *ENVIRONMENTS[name],
                f"cell={cell}",
                "mode=rtrl",
                f"+mode.wrapper.precondition_power={power}",
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
        f"{'step':>9} {'reward':>8} {'|td|':>8} {'value':>9} "
        f"{'zero':>7} {'<1e-12':>8} {'<1e-8':>8} {'p50gram':>10}"
    )
    for chunk in range(NUM_CHUNKS):
        state, logs = train(
            jax.random.split(chunk_keys[chunk], NUM_SEEDS), state, STEPS_PER_CHUNK
        )
        parts = influences_of(state.carry)
        flat = jnp.concatenate(
            [parameter_gram(p, cell).reshape(NUM_SEEDS, -1) for p in parts], axis=-1
        )

        def metric(key_name):
            value = logs.get(key_name)
            return float(jnp.nanmean(value)) if value is not None else float("nan")

        print(
            f"{(chunk + 1) * STEPS_PER_CHUNK:>9} "
            f"{metric('normalize_reward/mean'):>8.4f} "
            f"{metric('critic/absolute_td_error'):>8.4f} "
            f"{metric('critic/value'):>9.4f} "
            f"{float(jnp.mean(jnp.mean(flat == 0, axis=-1))):>7.3f} "
            f"{float(jnp.mean(jnp.mean(flat < 1e-12, axis=-1))):>8.3f} "
            f"{float(jnp.mean(jnp.mean(flat < 1e-8, axis=-1))):>8.3f} "
            f"{float(jnp.mean(jnp.percentile(flat, 50, axis=-1))):>10.2e}"
        )


if __name__ == "__main__":
    main()
