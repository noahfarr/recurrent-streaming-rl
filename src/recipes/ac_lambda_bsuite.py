import flax.linen as nn
from hydra.utils import instantiate
from streax.algorithms import RecurrentACLambda
from streax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
)
from streax.networks import sparse

from src.environments import environment
from src.networks import build_cell, heads
from src.networks.network import Network


def make(cfg):
    env, env_params = environment.make(**cfg.environment)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env, gamma=cfg.algorithm.gamma)
    feature_extractor = lambda obs, action, reward, done: nn.Sequential(
        (
            nn.Dense(features=128, kernel_init=sparse(sparsity=0.9)),
            nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-5),
            nn.leaky_relu,
            nn.Dense(features=128, kernel_init=sparse(sparsity=0.9)),
            nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-5),
            nn.leaky_relu,
        )
    )(obs)
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

    actor_optimizer = instantiate(cfg.actor_optimizer)
    critic_optimizer = instantiate(cfg.critic_optimizer)

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
