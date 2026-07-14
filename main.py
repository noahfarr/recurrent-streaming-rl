import json
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import lox
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf
from streamlet.loggers import MultiLogger

from src import algorithm
from src.utils import profile


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):

    num_steps = int(cfg.total_timesteps) // cfg.num_epochs

    config = OmegaConf.to_container(cfg, resolve=True)

    logger = MultiLogger(
        [
            instantiate(v, cfg=config, _recursive_=False, _convert_="all")
            for v in (cfg.loggers or {}).values()
        ]
    )

    agent = algorithm.make(cfg)

    init = jax.jit(jax.vmap(agent.init))
    train = profile(
        jax.jit(
            jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None)),
            static_argnums=(2,),
            donate_argnums=(1,),
        ),
        num_steps * cfg.num_seeds,
    )

    key = jax.random.key(cfg.seed)
    init_key, warmup_key, train_key = jax.random.split(key, 3)

    state = init(jax.random.split(init_key, cfg.num_seeds))

    learning_starts = int(cfg.get("learning_starts", 0) or 0)
    if learning_starts:
        warmup = jax.jit(
            jax.vmap(agent.warmup, in_axes=(0, 0, None)), static_argnums=(2,)
        )
        state = warmup(
            jax.random.split(warmup_key, cfg.num_seeds), state, learning_starts
        )

    train_keys = jax.random.split(train_key, cfg.num_epochs)
    cost = 0.0
    for epoch in range(cfg.num_epochs):
        (state, logs), SPS = train(
            jax.random.split(train_keys[epoch], cfg.num_seeds), state, num_steps
        )
        cost += (num_steps * cfg.num_seeds) / SPS

        mask = logs.pop("returned_episode").reshape(cfg.num_seeds, -1)
        returned_episode_returns = logs.pop("returned_episode_returns").reshape(
            cfg.num_seeds, -1
        )
        returned_episode_lengths = logs.pop("returned_episode_lengths").reshape(
            cfg.num_seeds, -1
        )
        returned_discounted_episode_returns = logs.pop(
            "returned_discounted_episode_returns"
        ).reshape(cfg.num_seeds, -1)

        episode_returns = [
            returned_episode_returns[i][mask[i]] for i in range(cfg.num_seeds)
        ]
        episode_lengths = [
            returned_episode_lengths[i][mask[i]] for i in range(cfg.num_seeds)
        ]
        discounted_episode_returns = [
            returned_discounted_episode_returns[i][mask[i]]
            for i in range(cfg.num_seeds)
        ]

        data = {
            "training/episode_returns": [r.mean(keepdims=True) for r in episode_returns],
            "training/episode_lengths": [x.mean(keepdims=True) for x in episode_lengths],
            "training/discounted_episode_returns": [
                r.mean(keepdims=True) for r in discounted_episode_returns
            ],
            "training/SPS": jnp.full((cfg.num_seeds, 1), SPS),
            **jax.tree.map(
                lambda x: jnp.nanmean(
                    x.reshape(cfg.num_seeds, -1), axis=1, keepdims=True
                ),
                logs,
            ),
        }

        steps = jnp.array([epoch, epoch + 1]) * num_steps
        logger.log(data, steps=steps)

        logger.log_artifact(
            algorithm.policy_params(state),
            epoch,
            metrics={"episode_returns": jnp.concatenate(episode_returns).mean()},
        )

    score = float(jnp.concatenate(episode_returns).mean())
    cost = float(cost)
    logger.log_summary(
        {
            "score": jnp.array([r.mean() for r in episode_returns]),
            "cost": jnp.full((cfg.num_seeds,), cost),
        }
    )
    logger.finish()

    result = {"score": score, "cost": cost}
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    (output_dir / "result.json").write_text(json.dumps(result))

    return result


if __name__ == "__main__":
    main()
