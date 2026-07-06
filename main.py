from functools import partial

import hydra
import jax
import jax.numpy as jnp
import lox
from hydra.utils import instantiate
from streax.loggers import MultiLogger
from omegaconf import OmegaConf

from src import algorithm
from src.utils import profile


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):

    num_steps = cfg.total_timesteps // cfg.num_epochs

    logger = MultiLogger(
        [
            instantiate(
                v,
                cfg=OmegaConf.to_container(cfg, resolve=True),
                _recursive_=False,
                _convert_="all",
            )
            for v in (cfg.loggers or {}).values()
        ]
    )

    agent = algorithm.make(cfg)

    init = jax.jit(jax.vmap(agent.init))
    train = profile(
        jax.jit(
            jax.vmap(
                lox.spool(agent.train),
                in_axes=(0, 0, None),
            ),
            static_argnums=(2,),
        ),
        num_steps * cfg.num_seeds,
    )

    key = jax.random.key(cfg.seed)
    init_key, train_key = jax.random.split(key)

    state = init(jax.random.split(init_key, cfg.num_seeds))

    def extract(data, prefix):
        returned_episode = data.pop("returned_episode")
        # Keep the leading per-seed axis, average over everything else (steps,
        # and for batched algorithms like PPO, num_envs too).
        reduce_axes = tuple(range(1, returned_episode.ndim))

        result = {}
        if returned_episode.any():
            for key, name in (
                ("returned_episode_returns", "episode_returns"),
                ("returned_episode_lengths", "episode_lengths"),
                ("returned_discounted_episode_returns", "discounted_episode_returns"),
            ):
                result[f"{prefix}/{name}"] = jnp.mean(
                    data.pop(key), where=returned_episode, axis=reduce_axes
                )
        if data:
            result.update(jax.vmap(lambda x: jax.tree.map(jnp.mean, x))(data))
        return result

    train_keys = jax.random.split(train_key, cfg.num_epochs)

    for epoch in range(cfg.num_epochs):
        (state, logs), SPS = train(
            jax.random.split(train_keys[epoch], cfg.num_seeds), state, num_steps
        )
        data = extract(logs, "training")
        data["training/SPS"] = SPS
        logger.log(data, steps=state.step.mean(dtype=int).item())

    logger.finish()


if __name__ == "__main__":
    main()
