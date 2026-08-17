import flax.linen as nn
import jax.numpy as jnp
from hydra.utils import instantiate
from streamlet.algorithms import RecurrentACLambda
from streamlet.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
)
from streamlet.networks import sparse

from src.environments import environment
from src.networks import Ravel, build_cell, heads, infer_feature_dim
from src.networks.feature_extractor import FeatureExtractor
from src.networks.flatten import Flatten
from src.networks.network import Network, SeparateActorCritic


def make(cfg):
    env, env_params = environment.make(**cfg.environment)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env, gamma=cfg.algorithm.gamma)

    num_actions = env.action_space(env_params).n

    def make_feature_extractor():
        return FeatureExtractor(
            observation_extractor=nn.Sequential(
                [
                    nn.Conv(
                        16,
                        kernel_size=(3, 3),
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
                    nn.Dense(128, kernel_init=sparse(sparsity=0.9)),
                    nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-5),
                    nn.leaky_relu,
                ]
            ),
        )

    feature_extractor = make_feature_extractor()

    feature_dim = infer_feature_dim(
        feature_extractor,
        jnp.zeros(env.observation_space(env_params).shape),
        jnp.zeros((), jnp.int32),
        jnp.zeros(()),
        jnp.zeros((), bool),
    )

    actor_head = heads.Categorical(
        action_dim=num_actions, kernel_init=sparse(sparsity=0.9)
    )
    critic_head = heads.VNetwork(kernel_init=sparse(sparsity=0.9))

    if cfg.network == "shared":
        network = Network(
            feature_extractor=feature_extractor,
            cell=build_cell(cfg, input_size=feature_dim),
            head=heads.ActorCritic(actor=actor_head, critic=critic_head),
        )
    else:
        network = SeparateActorCritic(
            actor=Network(
                feature_extractor=feature_extractor,
                cell=build_cell(cfg, input_size=feature_dim),
                head=actor_head,
            ),
            critic=Network(
                feature_extractor=make_feature_extractor(),
                cell=build_cell(cfg, input_size=feature_dim),
                head=critic_head,
            ),
        )

    network = Ravel(network=network)

    actor_optimizer = instantiate(cfg.actor_optimizer)
    critic_optimizer = instantiate(cfg.critic_optimizer)

    agent = RecurrentACLambda(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        network=network,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
    )
    return agent
