import flax.linen as nn
import jax.numpy as jnp
import optax
from hydra.utils import instantiate
from memorax.algorithms.ppo import PPO as MemoraxPPO
from memorax.environments.wrappers import (
    ClipActionWrapper,
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
)
from memorax.networks import FeatureExtractor, Network, heads
from memorax.networks.blocks import Projection, Stack

from src.algorithms.optimizers import inject_logger
from src.algorithms.ppo.ppo import PPO as LocalPPO
from src.environments import environment


def make(cfg):
    env, env_params = environment.make(**cfg.environment)
    env = ClipActionWrapper(env, low=-2.0, high=2.0)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env, gamma=cfg.algorithm.gamma)

    feature_extractor = FeatureExtractor(
        observation_extractor=nn.Sequential(
            [
                nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2))),
                nn.tanh,
            ]
        ),
    )

    torso = Stack(
        blocks=(
            instantiate(cfg.torso),
            Projection(
                features=64,
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
                bias_init=nn.initializers.constant(0.0),
                activation_fn=nn.tanh,
            ),
        )
    )

    actor_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.Gaussian(
            action_dim=env.action_space(env_params).shape[0],
            kernel_init=nn.initializers.orthogonal(0.01),
        ),
    )
    critic_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.VNetwork(
            kernel_init=nn.initializers.orthogonal(1.0),
        ),
    )

    def make_optimizer(prefix):
        return optax.chain(
            optax.clip_by_global_norm(0.5),
            inject_logger(optax.adam, prefix=prefix)(learning_rate=1e-4, eps=1e-5),
        )

    ppo_cls = MemoraxPPO if cfg.mode.name == "bptt" else LocalPPO

    return ppo_cls(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
        actor_optimizer=make_optimizer("actor_optimizer"),
        critic_optimizer=make_optimizer("critic_optimizer"),
    )
