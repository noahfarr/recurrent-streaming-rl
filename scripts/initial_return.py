import json
import sys
from functools import partial
from pathlib import Path

import jax
import lox
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig

from src import algorithm

CONFIG_DIR = "/home/farr/recurrent-streaming-rl/config"
OUTPUT = Path("/home/farr/recurrent-streaming-rl/paper/plots/popgym/initial_returns.json")

ENVIRONMENTS = {
    "RepeatPrevious": "popgymnax/repeat_previous/easy",
    "StatelessCartPole": "popgymnax/stateless_cartpole/easy",
    "CountRecall": "popgymnax/count_recall/easy",
    "Minesweeper": "popgymnax/minesweeper/easy",
    "Battleship": "popgymnax/battleship/easy",
    "NoisyStatelessCartPole": "popgymnax/noisy_stateless_cartpole/easy",
    "RepeatFirst": "popgymnax/repeat_first/easy",
    "Autoencode": "popgymnax/autoencode/easy",
    "HigherLower": "popgymnax/higher_lower/easy",
    "MultiArmedBandit": "popgymnax/multiarmed_bandit/easy",
    "Concentration": "popgymnax/concentration/easy",
}

ALGORITHMS = {
    "qrc": ["algorithm=qrc", "mode=rtrl"],
    "stream_q": ["algorithm=stream_q", "mode=rtrl"],
    "intentional_q": ["algorithm=intentional_q", "mode=rtrl", "q_optimizer.cfg.eta=0.1"],
    "ppo": ["algorithm=ppo", "mode=bptt", "algorithm.num_steps=256",
            "algorithm.num_minibatches=1", "algorithm.update_epochs=4"],
}

NUM_SEEDS = 10
NUM_STEPS = 20_000


def build(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=overrides, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
        return algorithm.make(cfg)


def episode_returns(logs):
    mask = np.asarray(logs["returned_episode"]).reshape(NUM_SEEDS, -1)
    values = np.asarray(logs["returned_episode_returns"]).reshape(NUM_SEEDS, -1)
    per_seed = [values[i][mask[i]] for i in range(NUM_SEEDS)]
    return np.array([r.mean() if r.size else np.nan for r in per_seed])


def behaviour(agent, state, keys, epsilon):
    def run(key, state, num_steps):
        def step(state, step_key):
            state, _ = agent.env_step(state, step_key, epsilon)
            return state, None

        state, _ = jax.lax.scan(step, state, jax.random.split(key, num_steps))
        return state

    spooled = jax.jit(jax.vmap(lox.spool(run), in_axes=(0, 0, None)), static_argnums=(2,))
    return episode_returns(spooled(keys, state, NUM_STEPS)[1])


def sampled(agent, state, keys):
    def run(key, state, num_steps):
        state, _ = jax.lax.scan(
            partial(agent.rollout, temperature=1.0),
            state,
            jax.random.split(key, num_steps),
        )
        return state

    spooled = jax.jit(jax.vmap(lox.spool(run), in_axes=(0, 0, None)), static_argnums=(2,))
    return episode_returns(spooled(keys, state, NUM_STEPS)[1])


def greedy(agent, state, keys):
    spooled = jax.jit(
        jax.vmap(lox.spool(agent.evaluate), in_axes=(0, 0, None)), static_argnums=(2,)
    )
    return episode_returns(spooled(keys, state, NUM_STEPS)[1])


def main():
    selected = sys.argv[1].split(",") if len(sys.argv) > 1 else list(ALGORITHMS)
    results = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    for task, environment in ENVIRONMENTS.items():
        for name, extra in ALGORITHMS.items():
            if name not in selected:
                continue
            overrides = [
                f"environment={environment}",
                "cell=rtu",
                f"num_seeds={NUM_SEEDS}",
                "total_timesteps=10_000_000",
                "num_epochs=50",
                *extra,
            ]
            agent = build(overrides)
            keys = jax.random.split(jax.random.key(0), NUM_SEEDS)
            state = jax.jit(jax.vmap(agent.init))(keys)
            eval_keys = jax.random.split(jax.random.key(1), NUM_SEEDS)

            greedy_values = greedy(agent, state, eval_keys)
            entry = {"greedy": float(np.nanmean(greedy_values)),
                     "greedy_seeds": [float(v) for v in greedy_values]}
            schedule = getattr(agent, "epsilon_schedule", None)
            if schedule is not None:
                epsilon = float(schedule(0))
                values = behaviour(agent, state, eval_keys, epsilon)
                entry["epsilon"] = epsilon
            else:
                values = sampled(agent, state, eval_keys)
            entry["behaviour"] = float(np.nanmean(values))
            entry["behaviour_seeds"] = [float(v) for v in values]
            results[f"{task}/{name}"] = entry
            print(f"{task:<24} {name:<15} greedy={entry['greedy']:+.4f} "
                  f"behaviour={entry['behaviour']:+.4f}", flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
