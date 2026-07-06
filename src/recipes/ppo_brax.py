import flax.linen as nn
import jax.numpy as jnp
import optax
from hydra.utils import instantiate
from streax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
)

from src.algorithms.optimizers import inject_logger
from src.algorithms.ppo.ppo import PPO
from src.environments import environment
from src.environments.wrappers import ClipActionWrapper
from src.networks import ObservationFeatureExtractor, build_cell, heads
from src.networks.network import Network


class ProjectedHead(nn.Module):
    features: int
    head: nn.Module

    @nn.compact
    def __call__(self, x, **kwargs):
        x = nn.Dense(
            self.features,
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.tanh(x)
        return self.head(x, **kwargs)


def make(cfg):
    env, env_params = environment.make(**cfg.environment)
    env = ClipActionWrapper(env, low=-2.0, high=2.0)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env, gamma=cfg.algorithm.gamma)

    feature_extractor = ObservationFeatureExtractor(
        layers=nn.Sequential(
            (
                nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2))),
                nn.tanh,
            )
        ),
    )

    cell = build_cell(cfg)

    actor_network = Network(
        feature_extractor=feature_extractor,
        cell=cell,
        head=ProjectedHead(
            features=64,
            head=heads.Gaussian(
                action_dim=env.action_space(env_params).shape[0],
                kernel_init=nn.initializers.orthogonal(0.01),
            ),
        ),
    )
    critic_network = Network(
        feature_extractor=feature_extractor,
        cell=cell,
        head=ProjectedHead(
            features=64,
            head=heads.VNetwork(kernel_init=nn.initializers.orthogonal(1.0)),
        ),
    )

    def make_optimizer(prefix):
        return optax.chain(
            optax.clip_by_global_norm(0.5),
            inject_logger(optax.adam, prefix=prefix)(learning_rate=1e-4, eps=1e-5),
        )

    return PPO(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
        actor_optimizer=make_optimizer("actor_optimizer"),
        critic_optimizer=make_optimizer("critic_optimizer"),
    )
