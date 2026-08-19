import flax.linen as nn
import jax.numpy as jnp
import optax
from hydra.utils import instantiate
from streamlet.algorithms import RecurrentQLambda
from streamlet.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
)
from streamlet.networks import sparse

from src.environments import environment
from src.networks import Ravel, build_cell, heads, infer_feature_dim
from src.networks.feature_extractor import FeatureExtractor
from src.networks.flatten import Flatten
from src.networks.network import Network


def make(cfg):
    assert cfg.num_seeds == 1, (
        "The ALE FFI broadcasts under vmap, so seeds must run as separate "
        f"processes; got num_seeds={cfg.num_seeds}"
    )

    env, env_params = environment.make(**cfg.environment)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env, gamma=cfg.algorithm.gamma)

    num_actions = env.action_space(env_params).n
    feature_extractor = FeatureExtractor(
        observation_extractor=nn.Sequential(
            [
                lambda x: jnp.moveaxis(x, -3, -1),
                nn.Conv(
                    32,
                    kernel_size=(8, 8),
                    strides=(5, 5),
                    padding="VALID",
                    kernel_init=sparse(sparsity=0.9),
                ),
                nn.LayerNorm(
                    use_bias=False,
                    use_scale=False,
                    epsilon=1e-5,
                    reduction_axes=(-3, -2, -1),
                ),
                nn.leaky_relu,
                nn.Conv(
                    64,
                    kernel_size=(4, 4),
                    strides=(3, 3),
                    padding="VALID",
                    kernel_init=sparse(sparsity=0.9),
                ),
                nn.LayerNorm(
                    use_bias=False,
                    use_scale=False,
                    epsilon=1e-5,
                    reduction_axes=(-3, -2, -1),
                ),
                nn.leaky_relu,
                nn.Conv(
                    64,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding="VALID",
                    kernel_init=sparse(sparsity=0.9),
                ),
                nn.LayerNorm(
                    use_bias=False,
                    use_scale=False,
                    epsilon=1e-5,
                    reduction_axes=(-3, -2, -1),
                ),
                nn.leaky_relu,
                Flatten(start_dim=-3, end_dim=-1),
                nn.Dense(256, kernel_init=sparse(sparsity=0.9)),
                nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-5),
                nn.leaky_relu,
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

    q_network = Ravel(
        network=Network(
            feature_extractor=feature_extractor,
            cell=cell,
            head=heads.QNetwork(
                action_dim=num_actions, kernel_init=sparse(sparsity=0.9)
            ),
        )
    )

    epsilon_schedule = optax.linear_schedule(
        cfg.epsilon_start,
        cfg.epsilon_end,
        int(cfg.total_timesteps * cfg.epsilon_fraction),
    )

    q_optimizer = instantiate(cfg.q_optimizer)

    agent = RecurrentQLambda(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        q_network=q_network,
        q_optimizer=q_optimizer,
        epsilon_schedule=epsilon_schedule,
    )
    return agent
