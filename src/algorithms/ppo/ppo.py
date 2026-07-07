from dataclasses import dataclass
from functools import partial
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import core, struct

from streamlet.utils import Timestep, Transition, canonicalize_dtype
from streamlet.utils.typing import (
    Array,
    Environment,
    EnvParams,
    EnvState,
    Key,
    PyTree,
)

from src.cells import RTRL
from src.utils.typing import Carry


def broadcast_done(done: Array, leaf: Array) -> Array:
    return done.reshape(done.shape + (1,) * (leaf.ndim - done.ndim))


@struct.dataclass(frozen=True)
class PPOConfig:
    num_envs: int
    num_steps: int
    gamma: float
    gae_lambda: float
    num_minibatches: int
    update_epochs: int
    normalize_advantage: bool
    clip_coefficient: float
    clip_value_loss: bool
    entropy_coefficient: float

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps


@struct.dataclass(frozen=True)
class PPOState:
    step: int
    update_step: int
    timestep: Timestep
    env_state: EnvState
    actor_params: core.FrozenDict[str, Any]
    actor_optimizer_state: optax.OptState
    actor_carry: Carry
    critic_params: core.FrozenDict[str, Any]
    critic_optimizer_state: optax.OptState
    critic_carry: Carry


@dataclass
class PPO:
    cfg: PPOConfig
    env: Environment
    env_params: EnvParams
    actor_network: nn.Module
    critic_network: nn.Module
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation

    def __post_init__(self):
        assert (
            self.cfg.update_epochs >= 1
        ), f"update_epochs ({self.cfg.update_epochs}) must be >= 1"
        assert self.cfg.batch_size % self.cfg.num_minibatches == 0, (
            f"num_envs * num_steps ({self.cfg.batch_size}) must be divisible by "
            f"num_minibatches ({self.cfg.num_minibatches})"
        )
        assert not isinstance(
            getattr(self.actor_network, "cell", None), RTRL
        ), "PPO requires a BPTT cell (mode=bptt), not RTRL"
        assert not isinstance(
            getattr(self.critic_network, "cell", None), RTRL
        ), "PPO requires a BPTT cell (mode=bptt), not RTRL"

    def apply(self, network: nn.Module, params, carry, timestep) -> tuple:
        return jax.vmap(network.apply, in_axes=(None, 0, 0, 0, 0, 0))(
            params, carry, *timestep
        )

    def generalized_advantage_estimation(self, carry: tuple, transition: Transition):
        advantage, next_value = carry
        delta = (
            transition.second.reward
            + self.cfg.gamma * (1 - transition.second.done) * next_value
            - transition.aux["value"]
        )
        advantage = (
            delta
            + self.cfg.gamma
            * self.cfg.gae_lambda
            * (1 - transition.second.done)
            * advantage
        )
        return (advantage, transition.aux["value"]), advantage

    def rollout(
        self, state: PPOState, key: Key, *, temperature: float
    ) -> tuple[PPOState, Transition]:
        action_key, step_key = jax.random.split(key)

        actor_carry, dist = self.apply(
            self.actor_network, state.actor_params, state.actor_carry, state.timestep
        )
        action, log_prob = dist.sample_and_log_prob(seed=action_key)
        action = jnp.where(temperature == 0.0, dist.mode(), action)

        critic_carry, value = self.apply(
            self.critic_network, state.critic_params, state.critic_carry, state.timestep
        )
        value = jnp.squeeze(value, axis=-1)

        step_keys = jax.random.split(step_key, self.cfg.num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, action, self.env_params)
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.bool_)
        del info

        transition = Transition(
            first=state.timestep,
            second=Timestep(obs=next_obs, action=action, reward=reward, done=done),
            aux={
                "log_prob": log_prob,
                "value": value,
            },
        )

        state = state.replace(
            step=state.step + self.cfg.num_envs,
            timestep=Timestep(
                obs=next_obs,
                action=jnp.where(
                    broadcast_done(done, action), jnp.zeros_like(action), action
                ),
                reward=jnp.where(done, jnp.zeros_like(reward), reward),
                done=done,
            ),
            env_state=env_state,
            actor_carry=actor_carry,
            critic_carry=critic_carry,
        )
        return state, transition

    def update_actor(
        self,
        state: PPOState,
        initial_carry: Carry,
        transitions: Transition,
    ) -> tuple[PPOState, Array, tuple[Array, Array, Array]]:
        advantages = transitions.aux["advantages"]
        if self.cfg.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        def actor_loss_fn(params: PyTree):
            _, probs = self.apply(
                self.actor_network, params, initial_carry, transitions.first
            )
            log_probs = probs.log_prob(transitions.second.action)
            entropy = probs.entropy().mean()
            ratio = jnp.exp(log_probs - transitions.aux["log_prob"])
            approximate_kl = jnp.mean(transitions.aux["log_prob"] - log_probs)
            clip_fraction = jnp.mean(
                (jnp.abs(ratio - 1.0) > self.cfg.clip_coefficient).astype(jnp.float32)
            )

            actor_loss = -jnp.minimum(
                ratio * advantages,
                jnp.clip(
                    ratio,
                    1.0 - self.cfg.clip_coefficient,
                    1.0 + self.cfg.clip_coefficient,
                )
                * advantages,
            ).mean()
            return actor_loss - self.cfg.entropy_coefficient * entropy, (
                entropy.mean(),
                approximate_kl.mean(),
                clip_fraction.mean(),
            )

        (actor_loss, aux), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(state.actor_params)
        lox.log({"actor/gradient_norm": optax.global_norm(actor_grads)})
        actor_updates, actor_optimizer_state = self.actor_optimizer.update(
            actor_grads, state.actor_optimizer_state, state.actor_params
        )
        actor_params = optax.apply_updates(state.actor_params, actor_updates)

        state = state.replace(
            actor_params=actor_params,
            actor_optimizer_state=actor_optimizer_state,
        )
        return state, actor_loss.mean(), aux

    def update_critic(
        self,
        state: PPOState,
        initial_carry: Carry,
        transitions: Transition,
    ) -> tuple[PPOState, Array]:
        returns = transitions.aux["returns"]

        def critic_loss_fn(params: PyTree):
            _, values = self.apply(
                self.critic_network, params, initial_carry, transitions.first
            )
            values = jnp.squeeze(values, axis=-1)

            critic_loss = jnp.square(values - returns)
            if self.cfg.clip_value_loss:
                clipped_value = transitions.aux["value"] + jnp.clip(
                    values - transitions.aux["value"],
                    -self.cfg.clip_coefficient,
                    self.cfg.clip_coefficient,
                )
                clipped_critic_loss = jnp.square(clipped_value - returns)
                critic_loss = jnp.maximum(critic_loss, clipped_critic_loss)
            return critic_loss.mean(), values

        (critic_loss, values), critic_grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(state.critic_params)
        explained_variance = 1 - jnp.var(returns - values) / jnp.var(returns)
        lox.log(
            {
                "critic/gradient_norm": optax.global_norm(critic_grads),
                "critic/explained_variance": explained_variance,
                "critic/value": values.mean(),
            }
        )
        critic_updates, critic_optimizer_state = self.critic_optimizer.update(
            critic_grads, state.critic_optimizer_state, state.critic_params
        )
        critic_params = optax.apply_updates(state.critic_params, critic_updates)

        state = state.replace(
            critic_params=critic_params, critic_optimizer_state=critic_optimizer_state
        )
        return state, critic_loss.mean()

    def update_minibatch(
        self, state: PPOState, minibatch: tuple
    ) -> tuple[PPOState, tuple[Array, Array, tuple[Array, Array, Array]]]:
        initial_actor_carry, initial_critic_carry, transitions = minibatch

        state, critic_loss = self.update_critic(
            state, initial_critic_carry, transitions
        )
        state, actor_loss, aux = self.update_actor(
            state, initial_actor_carry, transitions
        )

        return state, (actor_loss, critic_loss, aux)

    def update_epoch(self, carry: tuple, key: Key) -> tuple:
        state, initial_actor_carry, initial_critic_carry, transitions = carry

        if initial_actor_carry is not None:
            num_permutations = self.cfg.num_envs
            batch = (initial_actor_carry, initial_critic_carry, transitions)
        else:
            num_permutations = self.cfg.num_envs * self.cfg.num_steps
            batch = (
                initial_actor_carry,
                initial_critic_carry,
                jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), transitions),
            )

        permutation = jax.random.permutation(key, num_permutations)
        minibatches = jax.tree.map(
            lambda x: jnp.take(x, permutation, axis=0).reshape(
                self.cfg.num_minibatches, -1, *x.shape[1:]
            ),
            batch,
        )

        state, (
            actor_loss,
            critic_loss,
            (entropy, approximate_kl, clip_fraction),
        ) = jax.lax.scan(
            self.update_minibatch,
            state,
            minibatches,
        )

        metrics = jax.tree.map(
            lambda x: x.mean(),
            (actor_loss, critic_loss, entropy, approximate_kl, clip_fraction),
        )

        return (state, initial_actor_carry, initial_critic_carry, transitions), metrics

    def update_step(self, state: PPOState, key: Key) -> tuple[PPOState, None]:
        step_key, epoch_key = jax.random.split(key)

        initial_actor_carry = state.actor_carry
        initial_critic_carry = state.critic_carry

        step_keys = jax.random.split(step_key, self.cfg.num_steps)
        state, transitions = jax.lax.scan(
            partial(self.rollout, temperature=1.0),
            state,
            step_keys,
        )

        _, value = self.apply(
            self.critic_network, state.critic_params, state.critic_carry, state.timestep
        )
        value = jnp.squeeze(value, axis=-1)

        _, advantages = jax.lax.scan(
            self.generalized_advantage_estimation,
            (jnp.zeros_like(value), value),
            transitions,
            reverse=True,
            unroll=16,
        )
        returns = advantages + transitions.aux["value"]

        transitions = transitions.replace(
            aux={**transitions.aux, "advantages": advantages, "returns": returns}
        )
        transitions = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), transitions)

        epoch_keys = jax.random.split(epoch_key, self.cfg.update_epochs)
        (state, *_), metrics = jax.lax.scan(
            self.update_epoch,
            (state, initial_actor_carry, initial_critic_carry, transitions),
            epoch_keys,
        )

        actor_loss, critic_loss, entropy, approximate_kl, clip_fraction = jax.tree.map(
            lambda x: x.mean(), metrics
        )
        lox.log(
            {
                "actor/loss": actor_loss,
                "critic/loss": critic_loss,
                "actor/entropy": entropy,
                "actor/approximate_kl": approximate_kl,
                "actor/clip_fraction": clip_fraction,
            }
        )

        return state.replace(update_step=state.update_step + 1), None

    def init(self, key: Key) -> PPOState:
        env_key, actor_key, critic_key, actor_carry_key, critic_carry_key = (
            jax.random.split(key, 5)
        )

        env_keys = jax.random.split(env_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            (self.cfg.num_envs, *action_space.shape),
            dtype=canonicalize_dtype(action_space.dtype),
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        timestep = Timestep(obs=obs, action=action, reward=reward, done=done)

        actor_carry = self.actor_network.initialize_carry(actor_carry_key)
        critic_carry = self.critic_network.initialize_carry(critic_carry_key)
        actor_carry = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (self.cfg.num_envs, *x.shape)),
            actor_carry,
        )
        critic_carry = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (self.cfg.num_envs, *x.shape)),
            critic_carry,
        )

        actor_params = self.actor_network.init(
            actor_key, actor_carry, *jax.tree.map(lambda x: x[0], timestep)
        )
        critic_params = self.critic_network.init(
            critic_key, critic_carry, *jax.tree.map(lambda x: x[0], timestep)
        )

        actor_optimizer_state = self.actor_optimizer.init(actor_params)
        critic_optimizer_state = self.critic_optimizer.init(critic_params)

        return PPOState(
            step=0,
            update_step=0,
            timestep=timestep,
            actor_carry=actor_carry,
            critic_carry=critic_carry,
            env_state=env_state,
            actor_params=actor_params,
            critic_params=critic_params,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
        )

    def warmup(self, key: Key, state: PPOState, num_steps: int) -> PPOState:
        return state

    def train(self, key: Key, state: PPOState, num_steps: int) -> PPOState:
        keys = jax.random.split(
            key, num_steps // (self.cfg.num_steps * self.cfg.num_envs)
        )
        state, _ = jax.lax.scan(
            self.update_step,
            state,
            keys,
        )

        return state

    def evaluate(
        self,
        key: Key,
        state: PPOState,
        num_steps: int,
    ) -> PPOState:
        reset_key, actor_carry_key, critic_carry_key, eval_key = jax.random.split(
            key, 4
        )
        reset_keys = jax.random.split(reset_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_keys, self.env_params
        )
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            (self.cfg.num_envs, *action_space.shape),
            dtype=canonicalize_dtype(action_space.dtype),
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        timestep = Timestep(obs=obs, action=action, reward=reward, done=done)

        actor_carry = self.actor_network.initialize_carry(actor_carry_key)
        critic_carry = self.critic_network.initialize_carry(critic_carry_key)
        state = state.replace(
            timestep=timestep,
            actor_carry=jax.tree.map(
                lambda x: jnp.broadcast_to(x, (self.cfg.num_envs, *x.shape)),
                actor_carry,
            ),
            critic_carry=jax.tree.map(
                lambda x: jnp.broadcast_to(x, (self.cfg.num_envs, *x.shape)),
                critic_carry,
            ),
            env_state=env_state,
        )

        step_keys = jax.random.split(eval_key, num_steps)
        state, _ = jax.lax.scan(
            partial(self.rollout, temperature=0.0),
            state,
            step_keys,
        )

        return state
