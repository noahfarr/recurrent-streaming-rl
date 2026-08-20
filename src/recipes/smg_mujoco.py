import flax.linen as nn
import jax.numpy as jnp
from hydra.utils import instantiate
from streamlet.algorithms import SMGLambda
from streamlet.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
)
from streamlet.networks import sparse

from src.environments import environment
from src.environments.wrappers import ClipActionWrapper, TimeAwareObservationWrapper
from src.networks import build_cell, compute_dtype, heads, infer_feature_dim
from src.networks.network import Network


def make(cfg):
    env, env_params = environment.make(**cfg.environment)
    env = ClipActionWrapper(env)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env, gamma=cfg.algorithm.gamma)
    env = TimeAwareObservationWrapper(env, time_limit=1000)

    action_dim = env.action_space(env_params).shape[0]
    dtype = compute_dtype(cfg)

    def make_feature_extractor():
        return lambda obs, action, reward, done: nn.Sequential(
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
        )(jnp.concatenate([obs, action, jnp.atleast_1d(reward)]))

    feature_dim = infer_feature_dim(
        make_feature_extractor(),
        jnp.zeros(env.observation_space(env_params).shape),
        jnp.zeros((action_dim,)),
        jnp.zeros(()),
        jnp.zeros((), bool),
    )
    assert feature_dim == 128

    actor_network = Network(
        feature_extractor=make_feature_extractor(),
        cell=build_cell(cfg, input_size=feature_dim),
        head=heads.Projection(
            features=128,
            head=heads.SquashedGaussian(
                action_dim=action_dim,
                kernel_init=sparse(sparsity=0.9),
                dtype=dtype,
            ),
            dtype=dtype,
        ),
    )
    critic_network = Network(
        feature_extractor=make_feature_extractor(),
        cell=build_cell(cfg, input_size=feature_dim),
        head=heads.ContinuousQNetwork(kernel_init=sparse(sparsity=0.9), dtype=dtype),
    )

    agent = SMGLambda(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
        actor_optimizer=instantiate(cfg.actor_optimizer),
        critic_optimizer=instantiate(cfg.critic_optimizer),
    )
    return agent
