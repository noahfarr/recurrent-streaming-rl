import hydra
import jax
import jax.numpy as jnp
import lox
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
    for epoch in range(cfg.num_epochs):
        (state, logs), SPS = train(
            jax.random.split(train_keys[epoch], cfg.num_seeds), state, num_steps
        )

        mask = logs.pop("returned_episode")
        episode_returns = jnp.where(mask, logs.pop("returned_episode_returns"), jnp.nan)
        episode_lengths = jnp.where(mask, logs.pop("returned_episode_lengths"), jnp.nan)
        discounted_episode_returns = jnp.where(
            mask, logs.pop("returned_discounted_episode_returns"), jnp.nan
        )

        data = {
            "training/episode_returns": episode_returns,
            "training/episode_lengths": episode_lengths,
            "training/discounted_episode_returns": discounted_episode_returns,
            "training/SPS": jnp.full_like(episode_returns, SPS),
            **logs,
        }
        steps = jnp.array([epoch, epoch + 1]) * num_steps
        logger.log(data, steps=steps)

        logger.log_artifact(
            state.actor_params,
            epoch,
            metrics={"episode_returns": float(jnp.nanmean(episode_returns))},
        )

    logger.finish()


if __name__ == "__main__":
    main()
