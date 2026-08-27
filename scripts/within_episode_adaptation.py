import sys

import jax
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


def profile(agent, state, episodes, horizon, learning, epsilon):
    env_step = jax.jit(agent.env_step)
    update = jax.jit(agent.update_step)
    rewards = np.zeros((horizon,))
    counts = np.zeros((horizon,))
    key = jax.random.key(7)

    index = 0
    for _ in range(episodes * horizon):
        key, step_key = jax.random.split(key)
        state, transition = env_step(state, step_key, epsilon)
        if learning:
            state = update(state, transition)
        if index < horizon:
            rewards[index] += float(transition.second.reward)
            counts[index] += 1
        index += 1
        if bool(transition.second.done):
            index = 0
    return rewards / np.maximum(counts, 1)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    horizon = 51

    agent = build([
        "algorithm=stream_q",
        "cell=ffn",
        "mode=bptt",
        "environment=popgymnax/repeat_first/easy",
        "num_seeds=1",
        f"total_timesteps={steps}",
        "num_epochs=1",
    ])
    state = jax.jit(agent.init)(jax.random.key(0))
    train = jax.jit(lox.spool(agent.train), static_argnums=(2,))
    for index in range(4):
        state, _ = train(jax.random.key(index + 1), state, max(steps // 4, 1))

    for learning in [True, False]:
        curve = profile(agent, state, episodes, horizon, learning, 0.01)
        label = "learning ON " if learning else "learning OFF"
        early = curve[:5].mean()
        late = curve[-20:].mean()
        print(f"{label}  first5={early:+.4f}  last20={late:+.4f}  return={curve.sum():+.3f}")
        print("   " + " ".join(f"{curve[t]:+.2f}" for t in range(0, horizon, 5)))


if __name__ == "__main__":
    main()
