import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from hydra.utils import instantiate
from memorax.algorithms.ppo import PPO as MemoraxPPO
from memorax.networks import Network, heads

from src.algorithms.ppo.ppo import PPO as LocalPPO
from src.environments import environment
from src.environments.wrappers import (
    NormalizeVecObservation,
    NormalizeVecReward,
)


class FeatureExtractor(nn.Module):
    features: int
    num_actions: int

    @nn.compact
    def __call__(self, observation, action, reward, done, **kwargs):
        action_embedding = jax.nn.one_hot(action, num_classes=self.num_actions)
        x = jnp.concatenate([observation, action_embedding, reward], axis=-1)
        x = nn.Dense(
            self.features,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.tanh(x)
        return x, {}


class HeadMLP(nn.Module):
    hidden_features: int
    head: nn.Module

    @nn.compact
    def __call__(self, x, **kwargs):
        x = nn.Dense(
            self.hidden_features,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(x)
        x = nn.tanh(x)
        return self.head(x, **kwargs)

    def loss(self, *args, **kwargs):
        return self.head.loss(*args, **kwargs)


def make(cfg):
    env, env_params = environment.make(**cfg.environment)
    env = NormalizeVecObservation(env)
    env = NormalizeVecReward(env, cfg.algorithm.gamma)

    num_actions = env.action_space(env_params).n
    feature_extractor = FeatureExtractor(features=64, num_actions=num_actions)
    torso = instantiate(cfg.torso)
    actor_head = HeadMLP(
        hidden_features=64,
        head=heads.Categorical(
            action_dim=num_actions,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        ),
    )
    critic_head = HeadMLP(
        hidden_features=64,
        head=heads.VNetwork(
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        ),
    )
    actor_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=actor_head,
    )
    critic_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=critic_head,
    )

    def make_optimizer(learning_rate):
        return optax.chain(
            optax.clip_by_global_norm(1.5),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=learning_rate, eps=1e-5
            ),
        )

    ppo_cls = MemoraxPPO if cfg.mode.name == "bptt" else LocalPPO

    return ppo_cls(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
        actor_optimizer=make_optimizer(cfg.actor_lr),
        critic_optimizer=make_optimizer(cfg.critic_lr),
    )
