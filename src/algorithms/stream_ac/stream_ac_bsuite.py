import flax.linen as nn
from hydra.utils import instantiate
from memorax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
)
from memorax.networks import FeatureExtractor, Network, heads
from memorax.networks.initializers import sparse

from src.algorithms.stream_ac import StreamAC
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
                nn.Dense(features=128, kernel_init=sparse(sparsity=0.9)),
                nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-5),
                nn.leaky_relu,
            )
        ),
    )
    torso = instantiate(cfg.torso)
    actor_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.Categorical(
            action_dim=env.action_space(env_params).n,
            kernel_init=sparse(sparsity=0.9),
        ),
    )
    critic_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.VNetwork(kernel_init=sparse(sparsity=0.9)),
    )
    agent = StreamAC(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
    )
    return agent
