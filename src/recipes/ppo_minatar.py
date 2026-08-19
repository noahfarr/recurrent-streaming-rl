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
from src.networks import build_cell, compute_dtype, heads, infer_feature_dim
from src.networks.feature_extractor import FeatureExtractor
from src.networks.flatten import Flatten
from src.networks.network import Network


def make(cfg):
    env, env_params = environment.make(**cfg.environment)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env, gamma=cfg.algorithm.gamma)

    num_actions = env.action_space(env_params).n
    dtype = compute_dtype(cfg)
    feature_extractor = FeatureExtractor(
        observation_extractor=nn.Sequential(
            [
                lambda x: x.astype(dtype),
                nn.Conv(
                    16,
                    kernel_size=(3, 3),
                    padding="VALID",
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                    dtype=dtype,
                    param_dtype=jnp.float32,
                ),
                nn.relu,
                Flatten(start_dim=-3, end_dim=-1),
                nn.Dense(
                    128,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                    dtype=dtype,
                    param_dtype=jnp.float32,
                ),
                nn.relu,
            ]
        ),
        action_extractor=lambda action: jax.nn.one_hot(
            action, num_classes=num_actions, dtype=dtype
        ),
        reward_extractor=lambda reward: reward[..., None].astype(dtype),
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
        dtype=dtype,
    )
    critic_head = heads.VNetwork(
        kernel_init=orthogonal(1.0),
        bias_init=constant(0.0),
        dtype=dtype,
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
