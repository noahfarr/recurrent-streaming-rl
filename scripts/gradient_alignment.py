import sys

import jax
import jax.numpy as jnp
import lox
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig

from src import algorithm

CONFIG_DIR = "/home/farr/recurrent-streaming-rl/config"


def build(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=overrides, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
        return algorithm.make(cfg)


def cell_vector(tree):
    flat = jax.tree_util.tree_flatten_with_path(tree)[0]
    parts = [v.ravel() for k, v in flat if "cell" in jax.tree_util.keystr(k)]
    return jnp.concatenate(parts)


def main():
    length = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 500_000
    probes = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    base = [
        "algorithm=qrc",
        "environment=gymnax/bsuite/memory_chain",
        f"environment.env_params.max_steps_in_episode={length + 1}",
        "cell=min_gru",
        "algorithm.trace_lambda=1.0",
        "algorithm.gamma=0.999",
        "num_seeds=1",
        f"total_timesteps={steps}",
        "num_epochs=1",
    ]
    agent = build(base + ["mode=rtrl"])
    truncated = build(base + ["mode=bptt"])

    state = jax.jit(agent.init)(jax.random.key(0))
    train = jax.jit(lox.spool(agent.train), static_argnums=(2,))
    chunk = max(steps // 10, 1)
    for index in range(10):
        state, _ = train(jax.random.key(index + 1), state, chunk)
    print(f"trained {steps} steps at L={length}")

    exact_network, trunc_network = agent.q_network, truncated.q_network
    env_step = jax.jit(agent.env_step)

    cosines, ratios, retentions = [], [], []
    key = jax.random.key(99)
    for index in range(probes):
        key, step_key = jax.random.split(key)
        params, carry, timestep = state.params, state.q_carry, state.timestep
        hidden = carry.carry

        def exact(p):
            return exact_network.apply(p, carry, *timestep)[1].max()

        def trunc(p):
            return trunc_network.apply(p, hidden, *timestep)[1].max()

        e = cell_vector(jax.grad(exact)(params))
        t = cell_vector(jax.grad(trunc)(params))
        norm_e, norm_t = jnp.linalg.norm(e), jnp.linalg.norm(t)
        if norm_e > 1e-12 and norm_t > 1e-12:
            cosines.append(float(e @ t / (norm_e * norm_t)))
            ratios.append(float(norm_e / norm_t))

        jacobian = jax.jacrev(
            lambda h: trunc_network.apply(params, h, *timestep)[0]
        )(hidden)
        flat = jnp.reshape(jacobian, (hidden.size, hidden.size))
        retentions.append(float(jnp.max(jnp.abs(jnp.diag(flat)))))

        state, _ = env_step(state, step_key, 0.01)

    def summary(name, values):
        values = jnp.array(values)
        print(
            f"  {name:34s} mean={float(values.mean()):+.4f} "
            f"median={float(jnp.median(values)):+.4f} "
            f"min={float(values.min()):+.4f} max={float(values.max()):+.4f}"
        )

    print(f"over {len(cosines)} probe steps with identical hidden state:")
    summary("cosine(exact, TBPTT(1)) cell grad", cosines)
    summary("|exact| / |TBPTT(1)|", ratios)
    summary("rho = max |diag dh_t/dh_{t-1}|", retentions)
    rho = float(jnp.median(jnp.array(retentions)))
    print(f"\n  gamma*lambda = 0.9990   measured rho = {rho:.4f}   "
          f"dominant mode = {'trace' if 0.999 > rho else 'RTRL'}")


if __name__ == "__main__":
    main()
