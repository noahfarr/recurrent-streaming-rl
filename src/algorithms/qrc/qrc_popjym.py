import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from hydra.utils import instantiate
from memorax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
)
from memorax.networks import Network, heads
from memorax.networks.initializers import sparse

from src.algorithms.qrc import QRC
from src.environments import environment


class FeatureExtractor(nn.Module):
    features: int
    num_actions: int

    @nn.compact
    def __call__(self, observation, action, reward, done, **kwargs):
        action_embedding = jax.nn.one_hot(action, num_classes=self.num_actions)
        x = jnp.concatenate([observation, action_embedding, reward], axis=-1)
        x = nn.Dense(self.features, kernel_init=sparse(sparsity=0.9))(x)
        x = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-5)(x)
        x = nn.leaky_relu(x)
        return x, {}


class HeadMLP(nn.Module):
    hidden_features: int
    head: nn.Module

    @nn.compact
    def __call__(self, x, **kwargs):
        x = nn.Dense(self.hidden_features, kernel_init=sparse(sparsity=0.9))(x)
        x = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-5)(x)
        x = nn.leaky_relu(x)
        return self.head(x, **kwargs)

    def loss(self, *args, **kwargs):
        return self.head.loss(*args, **kwargs)


def make(cfg):
    env, env_params = environment.make(**cfg.environment)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env, gamma=cfg.algorithm.gamma)

    num_actions = env.action_space(env_params).n
    feature_extractor = FeatureExtractor(features=64, num_actions=num_actions)

    torso = instantiate(cfg.torso)

    q_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=HeadMLP(
            hidden_features=64,
            head=heads.DiscreteQNetwork(
                action_dim=num_actions, kernel_init=sparse(sparsity=0.9)
            ),
        ),
    )
    h_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=HeadMLP(
            hidden_features=64,
            head=heads.DiscreteQNetwork(
                action_dim=num_actions, kernel_init=sparse(sparsity=0.9)
            ),
        ),
    )

    epsilon_schedule = optax.linear_schedule(
        cfg.epsilon_start,
        cfg.epsilon_end,
        int(cfg.total_timesteps * cfg.epsilon_fraction),
    )

    q_optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(cfg.q_lr))
    h_optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(cfg.h_lr))

    agent = QRC(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        q_network=q_network,
        h_network=h_network,
        q_optimizer=q_optimizer,
        h_optimizer=h_optimizer,
        epsilon_schedule=epsilon_schedule,
    )
    return agent
