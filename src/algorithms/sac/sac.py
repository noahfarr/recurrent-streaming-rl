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
from src.utils.axes import broadcast_done
from src.utils.typing import Carry
from src.utils.update import periodic_incremental_update


@struct.dataclass(frozen=True)
class SACConfig:
    gamma: float
    tau: float
    train_frequency: int
    target_update_frequency: int
    target_entropy_scale: float
    batch_size: int
    buffer_size: int
    sequence_length: int
    gradient_steps: int = 1
    num_critics: int = 2


@struct.dataclass(frozen=True)
class SACState:
    step: int
    update_step: int
    timestep: Timestep
    env_state: EnvState
    buffer_state: Any
    actor_params: core.FrozenDict[str, Any]
    critic_params: core.FrozenDict[str, Any]
    critic_target_params: core.FrozenDict[str, Any]
    alpha_params: core.FrozenDict[str, Any]
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    alpha_optimizer_state: optax.OptState
    actor_carry: Carry
    critic_carry: Carry


@dataclass
class SAC:
    cfg: SACConfig
    env: Environment
    env_params: EnvParams
    actor_network: nn.Module
    critic_network: nn.Module
    alpha_network: nn.Module
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation
    alpha_optimizer: optax.GradientTransformation
    buffer: Any

    def __post_init__(self):
        assert (
            self.cfg.train_frequency >= 1
        ), f"train_frequency ({self.cfg.train_frequency}) must be >= 1"
        assert (
            self.cfg.gradient_steps >= 1
        ), f"gradient_steps ({self.cfg.gradient_steps}) must be >= 1"
        assert not isinstance(
            getattr(self.actor_network, "cell", None), RTRL
        ), "SAC requires a BPTT cell (mode=bptt), not RTRL"
        assert not isinstance(
            getattr(self.critic_network, "cell", None), RTRL
        ), "SAC requires a BPTT cell (mode=bptt), not RTRL"

    @property
    def target_entropy(self) -> float:
        action_dim = self.env.action_space(self.env_params).shape[0]
        return -self.cfg.target_entropy_scale * action_dim

    def rollout(
        self, state: SACState, key: Key, *, temperature: float
    ) -> tuple[SACState, Transition]:
        action_key, step_key = jax.random.split(key)
        actor_carry, dist = self.actor_network.apply(
            state.actor_params,
            state.actor_carry,
            *state.timestep,
            temperature=temperature,
        )
        action = dist.sample(seed=action_key)

        next_obs, env_state, reward, done, info = self.env.step(
            step_key, state.env_state, action, self.env_params
        )
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.bool_)
        del info

        transition = Transition(
            first=state.timestep,
            second=Timestep(obs=next_obs, action=action, reward=reward, done=done),
        )
        buffer_state = self.buffer.add(
            state.buffer_state, jax.tree.map(lambda x: x[None, None], transition)
        )

        state = state.replace(
            step=state.step + 1,
            timestep=Timestep(
                obs=next_obs,
                action=jnp.where(
                    broadcast_done(done, action), jnp.zeros_like(action), action
                ),
                reward=jnp.where(done, jnp.zeros_like(reward), reward),
                done=done,
            ),
            env_state=env_state,
            buffer_state=buffer_state,
            actor_carry=actor_carry,
        )
        return state, transition

    def update_critic(
        self, key, state, experience, actor_carry, critic_carry, target_carry
    ):
        first = experience.first
        second = experience.second

        alpha = jnp.exp(self.alpha_network.apply(state.alpha_params))

        _, next_dist = jax.vmap(
            self.actor_network.apply, in_axes=(None, 0, 0, 0, 0, 0)
        )(state.actor_params, actor_carry, *second)
        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=key)
        _, next_qs = jax.vmap(self.critic_network.apply, in_axes=(None, 0, 0, 0, 0, 0))(
            state.critic_target_params,
            target_carry,
            second.obs,
            next_actions,
            second.reward,
            second.done,
        )
        next_q = jnp.min(next_qs, axis=-1)
        next_value = next_q - alpha * next_log_probs
        target_q = jax.lax.stop_gradient(
            second.reward + self.cfg.gamma * (1 - second.done) * next_value
        )

        def critic_loss_fn(critic_params):
            _, qs = jax.vmap(self.critic_network.apply, in_axes=(None, 0, 0, 0, 0, 0))(
                critic_params,
                critic_carry,
                first.obs,
                second.action,
                first.reward,
                first.done,
            )
            td_error = qs - target_q[..., None]
            critic_loss = 0.5 * jnp.square(td_error).sum(axis=-1).mean()
            return critic_loss, qs

        (critic_loss, qs), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
            state.critic_params
        )
        updates, critic_optimizer_state = self.critic_optimizer.update(
            grads, state.critic_optimizer_state, state.critic_params
        )
        critic_params = optax.apply_updates(state.critic_params, updates)
        critic_target_params = periodic_incremental_update(
            critic_params,
            state.critic_target_params,
            state.update_step,
            self.cfg.target_update_frequency,
            self.cfg.tau,
        )

        state = state.replace(
            critic_params=critic_params,
            critic_target_params=critic_target_params,
            critic_optimizer_state=critic_optimizer_state,
        )
        lox.log({"critic/loss": critic_loss, "critic/q": qs.mean()})
        return state

    def update_actor(self, key, state, experience, actor_carry, critic_carry):
        first = experience.first

        alpha = jnp.exp(self.alpha_network.apply(state.alpha_params))

        def actor_loss_fn(actor_params):
            _, dist = jax.vmap(self.actor_network.apply, in_axes=(None, 0, 0, 0, 0, 0))(
                actor_params, actor_carry, *first
            )
            actions, log_probs = dist.sample_and_log_prob(seed=key)
            _, qs = jax.vmap(self.critic_network.apply, in_axes=(None, 0, 0, 0, 0, 0))(
                state.critic_params,
                critic_carry,
                first.obs,
                actions,
                first.reward,
                first.done,
            )
            q = jnp.min(qs, axis=-1)
            actor_loss = (alpha * log_probs - q).mean()
            return actor_loss, log_probs

        (actor_loss, log_probs), grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(state.actor_params)
        updates, actor_optimizer_state = self.actor_optimizer.update(
            grads, state.actor_optimizer_state, state.actor_params
        )
        actor_params = optax.apply_updates(state.actor_params, updates)

        state = state.replace(
            actor_params=actor_params, actor_optimizer_state=actor_optimizer_state
        )
        lox.log({"actor/loss": actor_loss, "actor/entropy": -log_probs.mean()})
        return state

    def update_alpha(self, key, state, experience, actor_carry):
        first = experience.first

        _, dist = jax.vmap(self.actor_network.apply, in_axes=(None, 0, 0, 0, 0, 0))(
            state.actor_params, actor_carry, *first
        )
        _, log_probs = dist.sample_and_log_prob(seed=key)

        def alpha_loss_fn(alpha_params):
            alpha = jnp.exp(self.alpha_network.apply(alpha_params))
            alpha_loss = (alpha * (-log_probs - self.target_entropy)).mean()
            return alpha_loss, alpha

        (alpha_loss, alpha), grads = jax.value_and_grad(alpha_loss_fn, has_aux=True)(
            state.alpha_params
        )
        updates, alpha_optimizer_state = self.alpha_optimizer.update(
            grads, state.alpha_optimizer_state, state.alpha_params
        )
        alpha_params = optax.apply_updates(state.alpha_params, updates)

        state = state.replace(
            alpha_params=alpha_params, alpha_optimizer_state=alpha_optimizer_state
        )
        lox.log({"alpha/loss": alpha_loss, "alpha/value": alpha})
        return state

    def update(self, key: Key, state: SACState) -> tuple[SACState, None]:
        sample_key, critic_key, actor_key, alpha_key = jax.random.split(key, 4)
        experience = self.buffer.sample(state.buffer_state, sample_key).experience

        actor_carry = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (self.cfg.batch_size, *x.shape)),
            self.actor_network.initialize_carry(jax.random.key(0)),
        )
        critic_carry = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (self.cfg.batch_size, *x.shape)),
            self.critic_network.initialize_carry(jax.random.key(0)),
        )
        target_carry = critic_carry

        state = self.update_critic(
            critic_key, state, experience, actor_carry, critic_carry, target_carry
        )
        state = self.update_actor(
            actor_key, state, experience, actor_carry, critic_carry
        )
        state = self.update_alpha(alpha_key, state, experience, actor_carry)
        return state, None

    def warmup(self, key: Key, state: SACState, num_steps: int) -> SACState:
        step_keys = jax.random.split(key, num_steps)
        state, _ = jax.lax.scan(
            partial(self.rollout, temperature=1.0), state, step_keys
        )
        return state

    def update_step(self, state: SACState, key: Key) -> tuple[SACState, None]:
        step_key, gradient_key = jax.random.split(key)

        step_keys = jax.random.split(step_key, self.cfg.train_frequency)
        state, _ = jax.lax.scan(
            partial(self.rollout, temperature=1.0), state, step_keys
        )

        gradient_keys = jax.random.split(gradient_key, self.cfg.gradient_steps)
        state, _ = jax.lax.scan(
            lambda state, key: self.update(key, state), state, gradient_keys
        )

        return state.replace(update_step=state.update_step + 1), None

    def init(self, key: Key) -> SACState:
        keys = jax.random.split(key, 6)
        env_key, actor_key, critic_key, alpha_key, actor_carry_key, critic_carry_key = (
            keys
        )

        obs, env_state = self.env.reset(env_key, self.env_params)
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            action_space.shape,
            dtype=canonicalize_dtype(action_space.dtype),
        )
        reward = jnp.zeros((), dtype=jnp.float32)
        done = jnp.ones((), dtype=jnp.bool_)
        timestep = Timestep(obs=obs, action=action, reward=reward, done=done)

        actor_carry = self.actor_network.initialize_carry(actor_carry_key)
        critic_carry = self.critic_network.initialize_carry(critic_carry_key)

        actor_params = self.actor_network.init(actor_key, actor_carry, *timestep)
        critic_params = self.critic_network.init(critic_key, critic_carry, *timestep)
        alpha_params = self.alpha_network.init(alpha_key)

        actor_optimizer_state = self.actor_optimizer.init(actor_params)
        critic_optimizer_state = self.critic_optimizer.init(critic_params)
        alpha_optimizer_state = self.alpha_optimizer.init(alpha_params)

        buffer_state = self.buffer.init(
            Transition(first=timestep, second=timestep)
        )

        return SACState(
            step=0,
            update_step=0,
            timestep=timestep,
            env_state=env_state,
            buffer_state=buffer_state,
            actor_params=actor_params,
            critic_params=critic_params,
            critic_target_params=critic_params,
            alpha_params=alpha_params,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
            alpha_optimizer_state=alpha_optimizer_state,
            actor_carry=actor_carry,
            critic_carry=critic_carry,
        )

    def train(self, key: Key, state: SACState, num_steps: int) -> SACState:
        keys = jax.random.split(key, num_steps // self.cfg.train_frequency)
        state, _ = jax.lax.scan(self.update_step, state, keys)
        return state

    def evaluate(self, key: Key, state: SACState, num_steps: int) -> SACState:
        reset_key, actor_carry_key, eval_key = jax.random.split(key, 3)
        obs, env_state = self.env.reset(reset_key, self.env_params)
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            action_space.shape,
            dtype=canonicalize_dtype(action_space.dtype),
        )
        reward = jnp.zeros((), dtype=jnp.float32)
        done = jnp.ones((), dtype=jnp.bool_)
        timestep = Timestep(obs=obs, action=action, reward=reward, done=done)

        actor_carry = self.actor_network.initialize_carry(actor_carry_key)
        state = state.replace(
            timestep=timestep, env_state=env_state, actor_carry=actor_carry
        )

        step_keys = jax.random.split(eval_key, num_steps)
        state, _ = jax.lax.scan(
            partial(self.rollout, temperature=0.0), state, step_keys
        )
        return state
