import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
from hydra.utils import instantiate
from streamlet.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
)

from src.algorithms.ppo.ppo import PPO
from src.environments import environment
from src.networks import build_cell, heads, infer_feature_dim
from src.networks.feature_extractor import FeatureExtractor
from src.networks.network import Network


def make(cfg):
    env, env_params = environment.make(**cfg.environment)

    num_actions = env.action_space(env_params).n
    feature_extractor = FeatureExtractor(
        observation_extractor=nn.Sequential(
            [
                nn.Dense(
                    64,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                ),
                nn.tanh,
            ]
        ),
    )
    feature_dim = infer_feature_dim(
        feature_extractor,
        jnp.zeros(env.observation_space(env_params).shape),
        jnp.zeros((), jnp.int32),
        jnp.zeros(()),
        jnp.zeros((), bool),
    )

    cell = build_cell(cfg, input_size=feature_dim)
    actor_head = heads.Categorical(
        action_dim=num_actions,
        kernel_init=orthogonal(0.01),
        bias_init=constant(0.0),
    )
    critic_head = heads.VNetwork(
        kernel_init=orthogonal(1.0),
        bias_init=constant(0.0),
    )
    actor_network = Network(
        feature_extractor=feature_extractor, cell=cell, head=actor_head
    )
    critic_network = Network(
        feature_extractor=feature_extractor, cell=cell, head=critic_head
    )

    return PPO(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
        actor_optimizer=instantiate(cfg.actor_optimizer),
        critic_optimizer=instantiate(cfg.critic_optimizer),
    )
