import flax.linen as nn
import jax.numpy as jnp
import optax
from hydra.utils import instantiate
from streamlet.algorithms import RecurrentQRCLambda
from streamlet.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
)
from streamlet.networks import sparse

from src.environments import environment
from src.networks import build_cell, compute_dtype, heads, infer_feature_dim
from src.networks.network import Network


def make(cfg):
    env, env_params = environment.make(**cfg.environment)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env, gamma=cfg.algorithm.gamma)
    dtype = compute_dtype(cfg)
    feature_extractor = lambda obs, action, reward, done: nn.Sequential(
        (
            lambda x: x.astype(dtype),
            nn.Dense(
                features=128,
                kernel_init=sparse(sparsity=0.9),
                dtype=dtype,
                param_dtype=jnp.float32,
            ),
            nn.LayerNorm(
                use_bias=False,
                use_scale=False,
                epsilon=1e-5,
                dtype=dtype,
                param_dtype=jnp.float32,
            ),
            nn.leaky_relu,
        )
    )(obs)
    feature_dim = infer_feature_dim(
        feature_extractor,
        jnp.zeros(env.observation_space(env_params).shape),
        jnp.zeros((), jnp.int32),
        jnp.zeros(()),
        jnp.zeros((), bool),
    )
    cell = build_cell(cfg, input_size=feature_dim)
    num_actions = env.action_space(env_params).n
    q_network = Network(
        feature_extractor=feature_extractor,
        cell=cell,
        head=heads.QNetwork(
            action_dim=num_actions,
            kernel_init=sparse(sparsity=0.9),
            dtype=dtype,
        ),
    )
    h_network = Network(
        feature_extractor=feature_extractor,
        cell=cell,
        head=heads.QNetwork(
            action_dim=num_actions,
            kernel_init=sparse(sparsity=0.9),
            dtype=dtype,
        ),
    )
    epsilon_schedule = optax.linear_schedule(
        cfg.epsilon_start, cfg.epsilon_end, int(cfg.total_timesteps * cfg.epsilon_fraction)
    )
    agent = RecurrentQRCLambda(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        q_network=q_network,
        h_network=h_network,
        q_optimizer=instantiate(cfg.q_optimizer),
        h_optimizer=instantiate(cfg.h_optimizer),
        epsilon_schedule=epsilon_schedule,
    )
    return agent
