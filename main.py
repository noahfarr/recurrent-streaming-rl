from functools import partial

import hydra
import jax
import jax.numpy as jnp
import lox
from hydra.utils import instantiate
from memorax.loggers import MultiLogger
from omegaconf import OmegaConf

from src import algorithm
from src.utils import profile, strip


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
                lox.spool(strip(agent.train)),
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
        info = data.pop("info")
        returned_episode = info["returned_episode"]

        result = {}
        if returned_episode.any():
            result[f"{prefix}/episode_returns"] = jnp.mean(
                info["returned_episode_returns"], where=returned_episode, axis=(1, 2)
            )
            result[f"{prefix}/episode_lengths"] = jnp.mean(
                info["returned_episode_lengths"], where=returned_episode, axis=(1, 2)
            )
            result[f"{prefix}/discounted_episode_returns"] = jnp.mean(
                info["returned_discounted_episode_returns"],
                where=returned_episode,
                axis=(1, 2),
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
        logger.log(data, step=state.step.mean(dtype=int).item())

    logger.finish()


if __name__ == "__main__":
    main()
