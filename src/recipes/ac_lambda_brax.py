import flax.linen as nn
import jax.numpy as jnp
from hydra.utils import instantiate
from streax.algorithms import RecurrentACLambda
from streax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
)
from streax.networks import sparse

from src.environments import environment
from src.environments.wrappers import ClipActionWrapper
from src.networks import build_cell, heads, infer_feature_dim
from src.networks.network import Network


class ProjectedHead(nn.Module):
    """Projects the torso output before the final head (post-cell block)."""

    features: int
    head: nn.Module

    @nn.compact
    def __call__(self, x, **kwargs):
        x = nn.Dense(
            self.features,
            kernel_init=sparse(sparsity=0.9),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.tanh(x)
        return self.head(x, **kwargs)


def make(cfg):
    env, env_params = environment.make(**cfg.environment)
    env = ClipActionWrapper(env)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env, gamma=cfg.algorithm.gamma)

    feature_extractor = lambda obs, action, reward, done: nn.Sequential(
        (
            nn.Dense(features=64, kernel_init=sparse(sparsity=0.9)),
            nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-5),
            nn.tanh,
        )
    )(obs)

    feature_dim = infer_feature_dim(
        feature_extractor,
        jnp.zeros(env.observation_space(env_params).shape),
        jnp.zeros(env.action_space(env_params).shape),
        jnp.zeros(()),
        jnp.zeros((), bool),
    )
    cell = build_cell(cfg, input_size=feature_dim)

    action_dim = env.action_space(env_params).shape[0]
    actor_network = Network(
        feature_extractor=feature_extractor,
        cell=cell,
        head=ProjectedHead(
            features=64,
            head=heads.Gaussian(
                action_dim=action_dim,
                kernel_init=nn.initializers.orthogonal(0.01),
                bias_init=nn.initializers.constant(0.0),
            ),
        ),
    )
    critic_network = Network(
        feature_extractor=feature_extractor,
        cell=cell,
        head=ProjectedHead(
            features=64,
            head=heads.VNetwork(
                kernel_init=nn.initializers.orthogonal(1.0),
                bias_init=nn.initializers.constant(0.0),
            ),
        ),
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
