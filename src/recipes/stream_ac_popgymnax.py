import flax.linen as nn
import jax
import jax.numpy as jnp
from hydra.utils import instantiate
from streamlet.algorithms import RecurrentACLambda
from streamlet.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
)
from streamlet.networks import sparse

from src.environments import environment
from src.networks import Ravel, build_cell, compute_dtype, heads, infer_feature_dim
from src.networks.feature_extractor import FeatureExtractor
from src.networks.network import Network, SeparateActorCritic


def make(cfg):
    env, env_params = environment.make(**cfg.environment)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env, gamma=cfg.algorithm.gamma)

    num_actions = env.action_space(env_params).n
    dtype = compute_dtype(cfg)

    def make_feature_extractor():
        return FeatureExtractor(
            observation_extractor=nn.Sequential(
                [
                    lambda x: x.astype(dtype),
                    nn.Dense(
                        64,
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
                ]
            ),
            action_extractor=lambda action: jax.nn.one_hot(
                action, num_classes=num_actions, dtype=dtype
            ),
            reward_extractor=lambda reward: reward[..., None].astype(dtype),
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
        action_dim=num_actions,
        kernel_init=sparse(sparsity=0.9),
        dtype=dtype,
    )
    critic_head = heads.VNetwork(kernel_init=sparse(sparsity=0.9), dtype=dtype)

    auxiliary_weight = cfg.auxiliary_loss_weight
    if cfg.network == "shared":
        head = heads.ActorCritic(actor=actor_head, critic=critic_head)
        if auxiliary_weight > 0:
            head = heads.AuxiliaryPrediction(head=head, features=num_actions)
        network = Network(
            feature_extractor=feature_extractor,
            cell=build_cell(cfg, input_size=feature_dim),
            head=head,
        )
    else:
        assert auxiliary_weight == 0, (
            "the auxiliary reward prediction head is wired for the shared network"
        )
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

    auxiliary_loss = None
    if auxiliary_weight > 0:

        def auxiliary_loss(transition):
            prediction = jax.tree.leaves(transition.aux["intermediates"])[0]
            target = transition.second.reward
            return auxiliary_weight * jnp.square(
                prediction[transition.second.action] - target
            )

    agent = RecurrentACLambda(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        network=network,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        auxiliary_loss=auxiliary_loss,
    )
    return agent
