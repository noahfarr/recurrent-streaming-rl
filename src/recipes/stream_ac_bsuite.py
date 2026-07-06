import flax.linen as nn
from hydra.utils import instantiate
from streax.algorithms import RecurrentACLambda
from streax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
)
from streax.networks import sparse
from streax.optimizers import ObGD, ObGDConfig

from src.environments import environment
from src.networks import ObservationFeatureExtractor, build_cell, heads
from src.networks.network import Network


def make(cfg):
    env, env_params = environment.make(**cfg.environment)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env, gamma=cfg.algorithm.gamma)
    feature_extractor = ObservationFeatureExtractor(
        layers=nn.Sequential(
            (
                nn.Dense(features=128, kernel_init=sparse(sparsity=0.9)),
                nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-5),
                nn.leaky_relu,
                nn.Dense(features=128, kernel_init=sparse(sparsity=0.9)),
                nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-5),
                nn.leaky_relu,
            )
        ),
    )
    cell = build_cell(cfg)
    actor_network = Network(
        feature_extractor=feature_extractor,
        cell=cell,
        head=heads.Categorical(
            action_dim=env.action_space(env_params).n,
            kernel_init=sparse(sparsity=0.9),
        ),
    )
    critic_network = Network(
        feature_extractor=feature_extractor,
        cell=cell,
        head=heads.VNetwork(kernel_init=sparse(sparsity=0.9)),
    )

    shared = dict(beta2=cfg.beta2, eps=cfg.eps, adaptive=cfg.adaptive)
    actor_optimizer = ObGD(
        cfg=ObGDConfig(lr=cfg.actor_lr, kappa=cfg.actor_kappa, **shared),
        name="actor_optimizer",
    )
    critic_optimizer = ObGD(
        cfg=ObGDConfig(lr=cfg.critic_lr, kappa=cfg.critic_kappa, **shared),
        name="critic_optimizer",
    )

    agent = RecurrentACLambda(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
    )
    return agent
