import flax.linen as nn
import jax.numpy as jnp
from flashbax.buffers.trajectory_buffer import make_trajectory_buffer
from hydra.utils import instantiate
from streamlet.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
)

from src.algorithms.sac import SAC
from src.environments import environment
from src.environments.wrappers import ClipActionWrapper
from src.networks import Network, build_cell, heads, infer_feature_dim
from src.networks.feature_extractor import FeatureExtractor


def make(cfg):
    env, env_params = environment.make(**cfg.environment)
    env = ClipActionWrapper(env, low=-1.0, high=1.0)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env, gamma=cfg.algorithm.gamma)

    feature_extractor = FeatureExtractor(
        observation_extractor=lambda obs: nn.Sequential(
            (
                nn.Dense(256, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2))),
                nn.relu,
            )
        )(obs),
        action_extractor=lambda action: action,
    )

    feature_dim = infer_feature_dim(
        feature_extractor,
        jnp.zeros(env.observation_space(env_params).shape),
        jnp.zeros(env.action_space(env_params).shape),
        jnp.zeros(()),
        jnp.zeros((), bool),
    )
    cell = build_cell(cfg, input_size=feature_dim)

    action_dim = env.action_space(env_params).shape[0]

    actor_network = Network(
        feature_extractor=feature_extractor,
        cell=cell,
        head=heads.Projection(
            features=256,
            head=heads.SquashedGaussian(
                action_dim=action_dim,
                kernel_init=nn.initializers.orthogonal(0.01),
            ),
        ),
    )

    num_critics = cfg.algorithm.num_critics

    critic_network = Network(
        feature_extractor=feature_extractor,
        cell=cell,
        head=lambda x, **kwargs: jnp.squeeze(
            nn.vmap(
                heads.QNetwork,
                variable_axes={"params": 0},
                split_rngs={"params": True},
                in_axes=None,
                out_axes=-1,
                axis_size=num_critics,
            )(action_dim=1, kernel_init=nn.initializers.orthogonal(1.0))(
                heads.Projection(features=256)(x)
            ),
            axis=-2,
        ),
    )

    alpha_network = heads.Alpha(initial_alpha=1.0)

    buffer = make_trajectory_buffer(
        add_batch_size=1,
        sample_batch_size=cfg.algorithm.batch_size,
        sample_sequence_length=cfg.algorithm.sequence_length,
        period=1,
        min_length_time_axis=cfg.algorithm.sequence_length,
        max_length_time_axis=cfg.algorithm.buffer_size,
    )

    return SAC(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
        alpha_network=alpha_network,
        actor_optimizer=instantiate(cfg.actor_optimizer),
        critic_optimizer=instantiate(cfg.critic_optimizer),
        alpha_optimizer=instantiate(cfg.alpha_optimizer),
        buffer=buffer,
    )
