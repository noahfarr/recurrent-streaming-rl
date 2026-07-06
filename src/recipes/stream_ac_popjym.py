import flax.linen as nn
import jax
import jax.numpy as jnp
from hydra.utils import instantiate
from streax.algorithms import RecurrentACLambda
from streax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
)
from streax.networks import sparse
from streax.optimizers import ObGD, ObGDConfig

from src.environments import environment
from src.networks import build_cell, heads
from src.networks.network import Network


class FeatureExtractor(nn.Module):
    features: int
    num_actions: int

    @nn.compact
    def __call__(self, observation, action, reward, done, **kwargs):
        action_embedding = jax.nn.one_hot(action, num_classes=self.num_actions)
        x = jnp.concatenate([observation, action_embedding, reward[None]], axis=-1)
        x = nn.Dense(self.features, kernel_init=sparse(sparsity=0.9))(x)
        x = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-5)(x)
        x = nn.leaky_relu(x)
        return x


class HeadMLP(nn.Module):
    hidden_features: int
    head: nn.Module

    @nn.compact
    def __call__(self, x, **kwargs):
        x = nn.Dense(self.hidden_features, kernel_init=sparse(sparsity=0.9))(x)
        x = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-5)(x)
        x = nn.leaky_relu(x)
        return self.head(x, **kwargs)


def make(cfg):
    env, env_params = environment.make(**cfg.environment)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env, gamma=cfg.algorithm.gamma)

    num_actions = env.action_space(env_params).n
    feature_extractor = FeatureExtractor(features=64, num_actions=num_actions)

    cell = build_cell(cfg)

    actor_network = Network(
        feature_extractor=feature_extractor,
        cell=cell,
        head=HeadMLP(
            hidden_features=64,
            head=heads.Categorical(
                action_dim=num_actions, kernel_init=sparse(sparsity=0.9)
            ),
        ),
    )
    critic_network = Network(
        feature_extractor=feature_extractor,
        cell=cell,
        head=HeadMLP(
            hidden_features=64,
            head=heads.VNetwork(kernel_init=sparse(sparsity=0.9)),
        ),
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
