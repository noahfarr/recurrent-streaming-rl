import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from hydra.utils import instantiate
from streax.algorithms import RecurrentQLambda
from streax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
)
from streax.networks import sparse

from src.environments import environment
from src.networks import build_cell, heads
from src.networks.feature_extractor import FeatureExtractor
from src.networks.network import Network


def make(cfg):
    env, env_params = environment.make(**cfg.environment)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env, gamma=cfg.algorithm.gamma)

    num_actions = env.action_space(env_params).n
    feature_extractor = FeatureExtractor(
        observation_extractor=nn.Sequential(
            [
                nn.Dense(64, kernel_init=sparse(sparsity=0.9)),
                nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-5),
                nn.leaky_relu,
            ]
        ),
        action_extractor=lambda action: jax.nn.one_hot(action, num_classes=num_actions),
        reward_extractor=lambda reward: reward[None],
    )

    feature_dim = jax.eval_shape(
        feature_extractor.init_with_output,
        jax.random.key(0),
        jnp.zeros(env.observation_space(env_params).shape),
        jnp.zeros((), jnp.int32),
        jnp.zeros(()),
        jnp.zeros((), bool),
    )[0].shape[-1]

    cell = build_cell(cfg, input_size=feature_dim)

    q_network = Network(
        feature_extractor=feature_extractor,
        cell=cell,
        head=heads.DiscreteQNetwork(
            action_dim=num_actions, kernel_init=sparse(sparsity=0.9)
        ),
    )

    epsilon_schedule = optax.linear_schedule(
        cfg.epsilon_start,
        cfg.epsilon_end,
        int(cfg.total_timesteps * cfg.epsilon_fraction),
    )

    make_optimizer = instantiate(cfg.optimizer)
    q_optimizer = make_optimizer(name="q_optimizer", lr=cfg.q_lr, kappa=cfg.q_kappa)

    agent = RecurrentQLambda(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        q_network=q_network,
        q_optimizer=q_optimizer,
        epsilon_schedule=epsilon_schedule,
    )
    return agent
