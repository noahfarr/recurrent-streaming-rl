from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import core, struct
from memorax.utils import Timestep, Transition
from memorax.utils.axes import (
    add_time_axis,
    remove_feature_axis,
    remove_time_axis,
)
from memorax.utils.typing import (
    Array,
    Carry,
    Discrete,
    Environment,
    EnvParams,
    EnvState,
    Key,
    PyTree,
)


from ..eligibility_trace import (
    Trace,
    TraceState,
    broadcast,
    trace_horizon_window,
)
from ..optimizers import OBGD, OBGDConfig, OBGDState


@struct.dataclass(frozen=True)
class StreamACConfig:
    num_envs: int
    gamma: float
    trace_lambda: float
    actor_lr: float
    critic_lr: float
    actor_kappa: float = 3.0
    critic_kappa: float = 2.0
    entropy_coefficient: float = 0.01
    adaptive: bool = False
    beta2: float = 0.999
    eps: float = 1e-8
    track: bool = False
    staleness_interval: int = 1
    taylor_trace: bool = False
    unroll: int = struct.field(pytree_node=False, default=2)


@struct.dataclass(frozen=True)
class StreamACState:
    step: int
    update_step: int
    timestep: Timestep
    actor_carry: Carry
    critic_carry: Carry
    env_state: EnvState
    actor_params: core.FrozenDict[str, Any]
    critic_params: core.FrozenDict[str, Any]
    actor_trace_state: TraceState
    critic_trace_state: TraceState
    actor_optimizer_state: OBGDState
    critic_optimizer_state: OBGDState


@dataclass
class StreamAC:
    cfg: StreamACConfig
    env: Environment
    env_params: EnvParams
    actor_network: nn.Module
    critic_network: nn.Module
    actor_trace: Trace = field(init=False)
    critic_trace: Trace = field(init=False)
    actor_optimizer: OBGD = field(init=False)
    critic_optimizer: OBGD = field(init=False)

    def __post_init__(self):
        bc = (
            min(
                trace_horizon_window(self.cfg.gamma, self.cfg.trace_lambda),
                self.env_params.max_steps_in_episode,
            )
            if self.cfg.track
            else 0
        )
        interval = self.cfg.staleness_interval
        self.actor_trace = Trace(buffer_capacity=bc, staleness_interval=interval)
        self.critic_trace = Trace(buffer_capacity=bc, staleness_interval=interval)

        shared = dict(
            beta2=self.cfg.beta2,
            eps=self.cfg.eps,
            adaptive=self.cfg.adaptive,
        )
        self.actor_optimizer = OBGD(
            cfg=OBGDConfig(
                lr=self.cfg.actor_lr, kappa=self.cfg.actor_kappa, **shared,
            ),
            name="actor_optimizer",
        )
        self.critic_optimizer = OBGD(
            cfg=OBGDConfig(
                lr=self.cfg.critic_lr, kappa=self.cfg.critic_kappa, **shared,
            ),
            name="critic_optimizer",
        )

    def _stochastic_action(
        self, key: Key, state: StreamACState
    ) -> tuple[StreamACState, Array, Array]:
        action_key, torso_key = jax.random.split(key)
        actor_carry, (dist, _) = self.actor_network.apply(
            state.actor_params,
            *state.timestep.to_sequence(),
            initial_carry=state.actor_carry,
            rngs={"torso": torso_key},
        )
        action, log_prob = dist.sample_and_log_prob(seed=action_key)
        action = remove_time_axis(action)
        log_prob = remove_time_axis(log_prob)
        return state.replace(actor_carry=actor_carry), action, log_prob

    def _deterministic_action(
        self, key: Key, state: StreamACState
    ) -> tuple[StreamACState, Array, Array]:
        actor_carry, (dist, _) = self.actor_network.apply(
            state.actor_params,
            *state.timestep.to_sequence(),
            initial_carry=state.actor_carry,
            rngs={"torso": key},
        )
        is_discrete = isinstance(self.env.action_space(self.env_params), Discrete)
        action = jnp.argmax(dist.logits, axis=-1) if is_discrete else dist.mode()
        log_prob = dist.log_prob(action)
        action = remove_time_axis(action)
        log_prob = remove_time_axis(log_prob)
        return state.replace(actor_carry=actor_carry), action, log_prob

    def _step(
        self, state: StreamACState, key: Key, *, policy: Callable
    ) -> tuple[StreamACState, Transition]:
        action_key, step_key = jax.random.split(key)
        actor_carry, critic_carry = state.actor_carry, state.critic_carry
        state, action, log_prob = policy(action_key, state)

        step_keys = jax.random.split(step_key, self.cfg.num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, action, self.env_params)
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.bool_)
        lox.log({"info": info})

        transition = Transition(
            first=state.timestep,
            second=Timestep(obs=next_obs, action=action, reward=reward, done=done),
            aux={
                "log_prob": log_prob,
                "actor_carry": actor_carry,
                "critic_carry": critic_carry,
            },
        )

        return (
            state.replace(
                step=state.step + self.cfg.num_envs,
                timestep=Timestep(
                    obs=next_obs,
                    action=jnp.where(done, jnp.zeros_like(action), action),
                    reward=jnp.where(done, jnp.zeros_like(reward), reward),
                    done=done,
                ),
                env_state=env_state,
            ),
            transition,
        )

    def _update_step(
        self, state: StreamACState, key: Key, *, policy: Callable
    ) -> tuple[StreamACState, None]:
        step_key, update_key = jax.random.split(key)
        state, transition = self._step(state, step_key, policy=policy)
        state = self._update(update_key, state, transition)
        return state.replace(update_step=state.update_step + 1), None

    def _update(
        self,
        key: Key,
        state: StreamACState,
        transition: Transition,
    ) -> StreamACState:
        actor_carry = transition.aux["actor_carry"]
        critic_carry = transition.aux["critic_carry"]
        action = transition.second.action

        def compute_state_value(params, timestep, carry, action):
            del action
            next_carry, (v, _) = self.critic_network.apply(
                params,
                *timestep.to_sequence(),
                initial_carry=carry,
                rngs={"torso": key},
            )
            return remove_feature_axis(remove_time_axis(v)), next_carry

        def compute_log_probs(params, timestep, carry, action):
            next_carry, (dist, _) = self.actor_network.apply(
                params,
                *timestep.to_sequence(),
                initial_carry=carry,
                rngs={"torso": key},
            )
            log_prob = remove_time_axis(dist.log_prob(add_time_axis(action)))
            entropy = remove_time_axis(dist.entropy())
            return (log_prob, entropy), next_carry

        def compute_td_error(params):
            next_carry, (v, _) = self.critic_network.apply(
                params,
                *transition.first.to_sequence(),
                initial_carry=critic_carry,
                rngs={"torso": key},
            )
            critic_value = remove_feature_axis(remove_time_axis(v))
            _, (next_v, _) = self.critic_network.apply(
                params,
                *transition.second.to_sequence(),
                initial_carry=next_carry,
                rngs={"torso": key},
            )
            next_value = remove_feature_axis(remove_time_axis(next_v))
            td_error = (
                transition.second.reward
                + self.cfg.gamma * (1.0 - transition.second.done) * next_value
                - critic_value
            )
            return critic_value, (td_error, next_carry)

        critic_values, critic_vjp, (td_error, next_critic_carry) = jax.vjp(
            compute_td_error, state.critic_params, has_aux=True
        )
        batch = self.cfg.num_envs
        (critic_grads,) = jax.vmap(critic_vjp)(jnp.eye(batch, dtype=critic_values.dtype))

        (log_probs, entropy_values), actor_vjp, _ = jax.vjp(
            partial(
                compute_log_probs,
                timestep=transition.first,
                carry=actor_carry,
                action=action,
            ),
            state.actor_params,
            has_aux=True,
        )
        eye = jnp.eye(batch, dtype=log_probs.dtype)
        zeros = jnp.zeros((batch, batch), dtype=log_probs.dtype)
        (log_prob_grads,) = jax.vmap(actor_vjp)((eye, zeros))
        (entropy_grads,) = jax.vmap(actor_vjp)((zeros, eye))

        reset_trace = transition.second.done
        discount = jnp.broadcast_to(
            jnp.float32(self.cfg.gamma * self.cfg.trace_lambda), reset_trace.shape
        )
        actor_grads = jax.tree.map(
            lambda lpg, eg: lpg
            + broadcast(jnp.sign(td_error), eg) * self.cfg.entropy_coefficient * eg,
            log_prob_grads,
            entropy_grads,
        )

        def actor_value_fn(params, timestep, carry, action, aux=None):
            (log_prob, entropy), next_carry = compute_log_probs(
                params, timestep, carry, action
            )
            sign = aux if aux is not None else jnp.sign(td_error)
            value = log_prob + broadcast(
                sign, log_prob
            ) * self.cfg.entropy_coefficient * entropy
            return value, next_carry

        actor_trace_state = self.actor_trace.update(
            state.actor_trace_state,
            actor_grads,
            discount,
            state.actor_params,
            partial(
                actor_value_fn,
                timestep=transition.first,
                carry=actor_carry,
                action=action,
            ),
        )
        actor_trace_state = self.actor_trace.observe(
            actor_trace_state,
            transition.first,
            action,
            reset_trace,
            actor_value_fn,
            state.actor_params,
            aux=jnp.sign(td_error),
        )
        critic_trace_state = self.critic_trace.update(
            state.critic_trace_state,
            critic_grads,
            discount,
            state.critic_params,
            partial(
                compute_state_value,
                timestep=transition.first,
                carry=critic_carry,
                action=action,
            ),
        )
        critic_trace_state = self.critic_trace.observe(
            critic_trace_state,
            transition.first,
            action,
            reset_trace,
            compute_state_value,
            state.critic_params,
        )

        actor_updates, actor_optimizer_state = self.actor_optimizer.update(
            state.actor_optimizer_state, actor_trace_state.taylor, td_error,
        )
        critic_updates, critic_optimizer_state = self.critic_optimizer.update(
            state.critic_optimizer_state, critic_trace_state.taylor, td_error,
        )

        actor_params = jax.tree.map(
            lambda p, u: p + u,
            state.actor_params,
            actor_updates,
        )
        critic_params = jax.tree.map(
            lambda p, u: p + u, state.critic_params, critic_updates
        )

        new_actor_trace_state = self.actor_trace.reset(actor_trace_state, reset_trace)
        new_critic_trace_state = self.critic_trace.reset(
            critic_trace_state, reset_trace
        )

        gamma_lambda = float(self.cfg.gamma * self.cfg.trace_lambda)
        target = critic_values + td_error
        explained_variance = 1 - jnp.var(td_error) / (jnp.var(target) + 1e-8)
        log_dict = {
            "critic/value": critic_values.mean(),
            "critic/td_error": td_error.mean(),
            "critic/explained_variance": explained_variance,
            "actor/log_prob": log_probs.mean(),
            "actor/entropy": entropy_values.mean(),
            **self.actor_trace.compute_staleness(
                new_actor_trace_state,
                actor_params,
                actor_value_fn,
                gamma_lambda,
                state.update_step,
                label="actor_trace",
            ),
            **self.critic_trace.compute_staleness(
                new_critic_trace_state,
                critic_params,
                compute_state_value,
                gamma_lambda,
                state.update_step,
                label="critic_trace",
            ),
        }
        log_dict |= {
            "actor_trace/trace_norm": optax.global_norm(
                new_actor_trace_state.trace
            ),
            "critic_trace/trace_norm": optax.global_norm(
                new_critic_trace_state.trace
            ),
        }
        if self.cfg.taylor_trace:
            log_dict |= {
                "actor_trace/drift_norm": optax.global_norm(
                    new_actor_trace_state.drift
                ),
                "actor_trace/correction_norm": optax.global_norm(
                    new_actor_trace_state.correction
                ),
                "critic_trace/drift_norm": optax.global_norm(
                    new_critic_trace_state.drift
                ),
                "critic_trace/correction_norm": optax.global_norm(
                    new_critic_trace_state.correction
                ),
            }
        lox.log(log_dict)

        new_state = dict(
            critic_carry=next_critic_carry,
            actor_params=actor_params,
            critic_params=critic_params,
            actor_trace_state=new_actor_trace_state,
            critic_trace_state=new_critic_trace_state,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
        )

        return state.replace(**new_state)

    def init(self, key: Key) -> StreamACState:
        env_key, actor_key, critic_key, torso_key = jax.random.split(key, 4)
        env_keys = jax.random.split(env_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            (self.cfg.num_envs, *action_space.shape), dtype=action_space.dtype
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        timestep = Timestep(obs=obs, action=action, reward=reward, done=done)
        actor_carry = self.actor_network.initialize_carry((self.cfg.num_envs, None))
        critic_carry = self.critic_network.initialize_carry((self.cfg.num_envs, None))
        actor_params = self.actor_network.init(
            {"params": actor_key, "torso": torso_key},
            *timestep.to_sequence(),
            initial_carry=actor_carry,
        )
        critic_params = self.critic_network.init(
            {"params": critic_key, "torso": torso_key},
            *timestep.to_sequence(),
            initial_carry=critic_carry,
        )

        actor_optimizer_state = self.actor_optimizer.init(
            actor_params, self.cfg.num_envs
        )
        critic_optimizer_state = self.critic_optimizer.init(
            critic_params, self.cfg.num_envs
        )

        actor_trace_state = self.actor_trace.init(
            actor_params,
            self.cfg.num_envs,
            taylor=self.cfg.taylor_trace,
            timestep=timestep,
            carry=actor_carry,
            action=action,
            aux=jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32),
        )
        critic_trace_state = self.critic_trace.init(
            critic_params,
            self.cfg.num_envs,
            taylor=self.cfg.taylor_trace,
            timestep=timestep,
            carry=critic_carry,
            action=action,
        )

        state = dict(
            step=0,
            update_step=0,
            timestep=timestep,
            actor_carry=actor_carry,
            critic_carry=critic_carry,
            env_state=env_state,
            actor_params=actor_params,
            critic_params=critic_params,
            actor_trace_state=actor_trace_state,
            critic_trace_state=critic_trace_state,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
        )

        return StreamACState(**state)

    def warmup(self, key: Key, state: StreamACState, num_steps: int) -> StreamACState:
        return state

    def train(self, key: Key, state: StreamACState, num_steps: int) -> StreamACState:
        keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        state, _ = jax.lax.scan(
            partial(self._update_step, policy=self._stochastic_action),
            state,
            keys,
            unroll=self.cfg.unroll,
        )
        return state

    def evaluate(
        self,
        key: Key,
        state: StreamACState,
        num_steps: int,
        deterministic: bool = True,
    ) -> StreamACState:
        reset_key, eval_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_keys, self.env_params
        )

        action_space = self.env.action_space(self.env_params)
        state = state.replace(
            step=0,
            timestep=Timestep(
                obs=obs,
                action=jnp.zeros(
                    (self.cfg.num_envs, *action_space.shape), dtype=action_space.dtype
                ),
                reward=jnp.zeros(self.cfg.num_envs),
                done=jnp.ones(self.cfg.num_envs, dtype=jnp.bool_),
            ),
            env_state=env_state,
            actor_carry=self.actor_network.initialize_carry(
                (self.cfg.num_envs, None)
            ),
            critic_carry=self.critic_network.initialize_carry(
                (self.cfg.num_envs, None)
            ),
        )

        policy = self._deterministic_action if deterministic else self._stochastic_action
        state, _ = jax.lax.scan(
            partial(self._step, policy=policy),
            state,
            jax.random.split(eval_key, num_steps),
        )
        return state
