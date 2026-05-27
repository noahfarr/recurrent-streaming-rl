import flax.linen as nn
import optax
from hydra.utils import instantiate
from memorax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
)
from memorax.networks import FeatureExtractor, Network, heads
from memorax.networks.initializers import sparse

from src.algorithms.qrc import QRC
from src.environments import environment


def make(cfg):
    env, env_params = environment.make(**cfg.environment)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env, gamma=cfg.algorithm.gamma)
    feature_extractor = FeatureExtractor(
        observation_extractor=nn.Sequential(
            (
                nn.Dense(features=128, kernel_init=sparse(sparsity=0.9)),
                nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-5),
                nn.leaky_relu,
            )
        ),
    )
    torso = instantiate(cfg.torso)
    num_actions = env.action_space(env_params).n
    q_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.DiscreteQNetwork(
            action_dim=num_actions, kernel_init=sparse(sparsity=0.9)
        ),
    )
    h_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.DiscreteQNetwork(
            action_dim=num_actions, kernel_init=sparse(sparsity=0.9)
        ),
    )
    epsilon_schedule = optax.linear_schedule(
        cfg.epsilon_start, cfg.epsilon_end, int(cfg.total_timesteps * cfg.epsilon_fraction)
    )
    agent = QRC(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        q_network=q_network,
        h_network=h_network,
        q_optimizer=optax.sgd(cfg.q_lr),
        h_optimizer=optax.sgd(cfg.h_lr),
        epsilon_schedule=epsilon_schedule,
    )
    return agent
