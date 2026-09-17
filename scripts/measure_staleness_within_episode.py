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


def main():
    length = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000

    agent = build([
        "algorithm=qrc",
        "environment=gymnax/bsuite/memory_chain",
        f"environment.env_params.max_steps_in_episode={length + 1}",
        "cell=min_gru", "mode=rtrl",
        "algorithm.trace_lambda=1.0", "algorithm.gamma=0.999",
        f"q_optimizer.lr={learning_rate}",
        "num_seeds=1", f"total_timesteps={steps}", "num_epochs=1",
    ])
    state = jax.jit(agent.init)(jax.random.key(0))
    train = jax.jit(lox.spool(agent.train), static_argnums=(2,))
    for index in range(4):
        state, _ = train(jax.random.key(index + 1), state, max(steps // 4, 1))

    env_step = jax.jit(agent.env_step)
    update = jax.jit(agent.update_step)

    key = jax.random.key(99)
    for _ in range(length + 2):
        key, step_key = jax.random.split(key)
        state, transition = env_step(state, step_key, 0.01)
        state = update(state, transition)
        if bool(transition.second.done):
            break

    start_carry = state.q_carry
    timesteps, stale_influences, drifts = [], [], []
    initial_parameters = state.params
    for _ in range(length):
        timesteps.append(state.timestep)
        key, step_key = jax.random.split(key)
        state, transition = env_step(state, step_key, 0.01)
        state = update(state, transition)
        stale_influences.append(influence_of(state.q_carry))
        drifts.append(float(jnp.linalg.norm(
            jax.flatten_util.ravel_pytree(state.params)[0]
            - jax.flatten_util.ravel_pytree(initial_parameters)[0])))

    final_parameters = state.params
    carry = start_carry
    errors = []
    for index, timestep in enumerate(timesteps):
        carry, _ = agent.q_network.apply(final_parameters, carry, *timestep)
        ideal = influence_of(carry)
        errors.append(float(jnp.linalg.norm(ideal - stale_influences[index])))

    lags = np.arange(1, len(errors) + 1, dtype=float)
    errors = np.array(errors)
    drifts = np.array(drifts)

    print(f"within-episode staleness, L={length}, lr={learning_rate}")
    print(f"{'t':>5} {'E_t':>12} {'drift':>12} {'E_t/t^2':>12}")
    for t in [1, 2, 4, 8, 16, 32, 64, len(errors)]:
        if t > len(errors):
            continue
        print(f"{t:5d} {errors[t-1]:12.4e} {drifts[t-1]:12.4e} {errors[t-1]/t**2:12.4e}")

    mask = errors > 0
    if mask.sum() >= 5:
        slope = np.polyfit(np.log(lags[mask]), np.log(errors[mask]), 1)[0]
        drift_slope = np.polyfit(np.log(lags[mask]), np.log(drifts[mask] + 1e-20), 1)[0]
        print(f"\n  fitted  E_t ~ t^{slope:.2f}     (bound predicts 2 at rho=1)")
        print(f"  fitted  drift ~ t^{drift_slope:.2f}  (assumption A2 predicts 1)")


if __name__ == "__main__":
    main()
