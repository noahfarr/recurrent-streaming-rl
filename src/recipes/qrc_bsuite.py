import flax.linen as nn
import optax
from hydra.utils import instantiate
from streax.algorithms import RecurrentQRCLambda
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
        )
    )(obs)
    cell = build_cell(cfg)
    num_actions = env.action_space(env_params).n
    q_network = Network(
        feature_extractor=feature_extractor,
        cell=cell,
        head=heads.DiscreteQNetwork(
            action_dim=num_actions, kernel_init=sparse(sparsity=0.9)
        ),
    )
    h_network = Network(
        feature_extractor=feature_extractor,
        cell=cell,
        head=heads.DiscreteQNetwork(
            action_dim=num_actions, kernel_init=sparse(sparsity=0.9)
        ),
    )
    epsilon_schedule = optax.linear_schedule(
        cfg.epsilon_start, cfg.epsilon_end, int(cfg.total_timesteps * cfg.epsilon_fraction)
    )
    make_optimizer = instantiate(cfg.optimizer)
    agent = RecurrentQRCLambda(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        q_network=q_network,
        h_network=h_network,
        q_optimizer=make_optimizer(name="q_optimizer", lr=cfg.q_lr),
        h_optimizer=make_optimizer(name="h_optimizer", lr=cfg.h_lr),
        epsilon_schedule=epsilon_schedule,
    )
    return agent
