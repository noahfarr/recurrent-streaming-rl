import sys

import jax
import jax.numpy as jnp
import lox
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig

from src import algorithm

CONFIG_DIR = "/home/farr/recurrent-streaming-rl/config"


def build(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=overrides, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
        return algorithm.make(cfg)


def influence_of(carry):
    if hasattr(carry, "influence"):
        return carry.influence
    if hasattr(carry, "carry"):
        return influence_of(carry.carry)
    return None


def measure(length, learning_rate, steps=100_000, seed=0):
    base = [
        "algorithm=qrc",
        "environment=gymnax/bsuite/memory_chain",
        f"environment.env_params.max_steps_in_episode={length + 1}",
        "cell=min_gru",
        "mode=rtrl",
        "algorithm.trace_lambda=1.0",
        "algorithm.gamma=0.999",
        f"q_optimizer.lr={learning_rate}",
        "num_seeds=1",
        f"total_timesteps={steps}",
        "num_epochs=1",
    ]
    agent = build(base)
    state = jax.jit(agent.init)(jax.random.key(seed))
    train = jax.jit(lox.spool(agent.train), static_argnums=(2,))
    for index in range(4):
        state, _ = train(jax.random.key(index + 1), state, max(steps // 4, 1))

    env_step = jax.jit(agent.env_step)
    update = jax.jit(agent.update_step)

    # advance to an episode boundary so the replay starts from a fresh carry
    key = jax.random.key(1234)
    for _ in range(length + 2):
        key, step_key = jax.random.split(key)
        state, transition = env_step(state, step_key, 0.01)
        state = update(state, transition)
        if bool(transition.second.done):
            break

    # run one episode, continuing to update, recording the inputs
    timesteps, start_carry = [], state.q_carry
    for _ in range(length + 1):
        timesteps.append(state.timestep)
        key, step_key = jax.random.split(key)
        state, transition = env_step(state, step_key, 0.01)
        state = update(state, transition)

    stale = influence_of(state.q_carry)
    final_parameters = state.params

    # replay the identical sequence with the parameters frozen at their final value
    carry = start_carry
    for timestep in timesteps:
        carry, _ = agent.q_network.apply(final_parameters, carry, *timestep)
    ideal = influence_of(carry)

    error = float(jnp.linalg.norm(ideal - stale))
    scale = float(jnp.linalg.norm(stale))
    retention = float(jnp.median(jnp.abs(jnp.diag(jnp.reshape(
        jax.jacrev(lambda h: agent.q_network.apply(final_parameters, carry.replace(carry=h), *timesteps[-1])[0].carry)(carry.carry),
        (carry.carry.size, carry.carry.size))))))
    return error, scale, retention


def main():
    learning_rate = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-4
    lengths = [int(v) for v in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["8", "16", "32", "64"])]
    print(f"staleness scaling, lr={learning_rate}")
    print(f"{'L':>5} {'E_T':>12} {'|S|':>12} {'E_T/L^2':>12} {'E_T/L':>12}")
    rows = []
    for length in lengths:
        error, scale, retention = measure(length, learning_rate)
        rows.append((length, error))
        print(f"{length:5d} {error:12.4e} {scale:12.4e} {error / length**2:12.4e} {error / length:12.4e}")
    lengths_array = np.array([r[0] for r in rows], dtype=float)
    errors = np.array([r[1] for r in rows], dtype=float)
    mask = errors > 0
    if mask.sum() >= 2:
        slope = np.polyfit(np.log(lengths_array[mask]), np.log(errors[mask]), 1)[0]
        print(f"\n  fitted exponent  E_T ~ L^{slope:.2f}   (bound predicts 2 at rho=1)")


if __name__ == "__main__":
    main()
