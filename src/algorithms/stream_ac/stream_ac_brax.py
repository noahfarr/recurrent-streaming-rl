import flax.linen as nn
from hydra.utils import instantiate
from memorax.environments.wrappers import (
    ClipActionWrapper,
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
)
from memorax.networks import FeatureExtractor, Network, heads
from memorax.networks.blocks import Projection, Stack
from memorax.networks.initializers import sparse

from src.algorithms.stream_ac import StreamAC
from src.environments import environment


def make(cfg):
    env, env_params = environment.make(**cfg.environment)
    env = ClipActionWrapper(env)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env, gamma=cfg.algorithm.gamma)

    feature_extractor = FeatureExtractor(
        observation_extractor=nn.Sequential(
            (
                nn.Dense(features=64, kernel_init=sparse(sparsity=0.9)),
                nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-5),
                nn.tanh,
            )
        ),
    )

    torso = Stack(
        blocks=(
            instantiate(cfg.torso),
            Projection(
                features=64,
                kernel_init=sparse(sparsity=0.9),
                bias_init=nn.initializers.constant(0.0),
                activation_fn=nn.tanh,
            ),
        )
    )

    action_dim = env.action_space(env_params).shape[0]
    actor_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.Gaussian(
            action_dim=action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0),
        ),
    )
    critic_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.VNetwork(
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.constant(0.0),
        ),
    )

    agent = StreamAC(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
    )
    return agent
