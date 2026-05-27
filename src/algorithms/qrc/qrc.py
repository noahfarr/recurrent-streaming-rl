import math
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
from memorax.utils.axes import add_feature_axis, remove_feature_axis, remove_time_axis
from memorax.utils.typing import (
    Array,
    Carry,
    Environment,
    EnvParams,
    EnvState,
    Key,
    PyTree,
)


from ..eligibility_trace import Trace, TraceState, trace_horizon_window


def broadcast(x: Array, y: Array) -> Array:
    return x[(slice(None),) + (None,) * (y.ndim - 1)]


@struct.dataclass(frozen=True)
class QRCConfig:
    num_envs: int
    gamma: float
    trace_lambda: float
    gradient_correction: bool
    regularization_coefficient: float
    track: bool = False
    staleness_interval: int = 1
    taylor_trace: bool = False
    unroll: int = struct.field(pytree_node=False, default=2)


@struct.dataclass(frozen=True)
class QRCState:
    step: int
    update_step: int
    timestep: Timestep
    q_carry: Carry
    h_carry: Carry
    env_state: EnvState
    q_params: core.FrozenDict[str, Any]
    h_params: core.FrozenDict[str, Any]
    q_optimizer_state: optax.OptState
    h_optimizer_state: optax.OptState
    q_trace_state: TraceState
    h_trace_state: TraceState
    bias_trace_state: TraceState


@dataclass
class QRC:
    cfg: QRCConfig
    env: Environment
    env_params: EnvParams
    q_network: nn.Module
    h_network: nn.Module
    q_optimizer: optax.GradientTransformation
    h_optimizer: optax.GradientTransformation
    epsilon_schedule: Callable
    q_trace: Trace = field(init=False)
    h_trace: Trace = field(init=False)
    bias_trace: Trace = field(init=False)

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
        self.q_trace = Trace(buffer_capacity=bc, staleness_interval=interval)
        self.h_trace = Trace(buffer_capacity=bc, staleness_interval=interval)
        self.bias_trace = Trace(buffer_capacity=0)

    def _greedy_action(
        self, key: Key, state: QRCState
    ) -> tuple[QRCState, Array, Array]:
        q_carry, (q_values, _) = self.q_network.apply(
            state.q_params,
            *state.timestep.to_sequence(),
            initial_carry=state.q_carry,
            rngs={"torso": key},
        )
        q_values = remove_time_axis(q_values)
        action = jnp.argmax(q_values, axis=-1)
        return (
            state.replace(q_carry=q_carry),
            action,
            jnp.zeros(self.cfg.num_envs, dtype=jnp.bool_),
        )

    def _random_action(
        self, key: Key, state: QRCState
    ) -> tuple[QRCState, Array, Array]:
        action_space = self.env.action_space(self.env_params)
        action = jax.random.randint(
            key,
            (self.cfg.num_envs, *action_space.shape),
            minval=0,
            maxval=action_space.n,
        )
        return state, action, jnp.ones(self.cfg.num_envs, dtype=jnp.bool_)

    def _epsilon_greedy_action(
        self, key: Key, state: QRCState
    ) -> tuple[QRCState, Array, Array]:
        random_key, greedy_key, sample_key = jax.random.split(key, 3)
        state, random_action, _ = self._random_action(random_key, state)
        state, greedy_action, _ = self._greedy_action(greedy_key, state)

        epsilon = self.epsilon_schedule(state.step)
        is_random = jax.random.uniform(sample_key, (self.cfg.num_envs,)) < epsilon
        action = jnp.where(
            broadcast(is_random, greedy_action), random_action, greedy_action
        )
        non_greedy = is_random & (random_action != greedy_action)
        return state, action, non_greedy

    def _step(
        self, state: QRCState, key: Key, *, policy: Callable
    ) -> tuple[QRCState, Transition]:
        action_key, step_key = jax.random.split(key)
        q_carry, h_carry = state.q_carry, state.h_carry
        state, action, non_greedy = policy(action_key, state)

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
            aux={"non_greedy": non_greedy, "q_carry": q_carry, "h_carry": h_carry},
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
        self, state: QRCState, key: Key, *, policy: Callable
    ) -> tuple[QRCState, None]:
        step_key, update_key = jax.random.split(key)
        state, transition = self._step(state, step_key, policy=policy)
        state = self._update(update_key, state, transition)
        return state.replace(update_step=state.update_step + 1), None

    def _update(
        self,
        key: Key,
        state: QRCState,
        transition: Transition,
    ) -> QRCState:
        q_carry = transition.aux["q_carry"]
        h_carry = transition.aux["h_carry"]

        action = transition.second.action
        non_greedy = transition.aux["non_greedy"]

        def compute_q_value(params, timestep, carry, action):
            next_carry, (q_values, _) = self.q_network.apply(
                params,
                *timestep.to_sequence(),
                initial_carry=carry,
                rngs={"torso": key},
            )
            value = remove_feature_axis(
                jnp.take_along_axis(
                    remove_time_axis(q_values),
                    add_feature_axis(action),
                    axis=-1,
                )
            )
            return value, next_carry

        def compute_h_value(params, timestep, carry, action):
            next_carry, (h_values, _) = self.h_network.apply(
                params,
                *timestep.to_sequence(),
                initial_carry=carry,
                rngs={"torso": key},
            )
            value = remove_feature_axis(
                jnp.take_along_axis(
                    remove_time_axis(h_values),
                    add_feature_axis(action),
                    axis=-1,
                )
            )
            return value, next_carry

        def compute_td_error(params):
            (next_q_carry, (q_values, _)) = self.q_network.apply(
                params,
                *transition.first.to_sequence(),
                initial_carry=q_carry,
                rngs={"torso": key},
            )
            q_value = remove_feature_axis(
                jnp.take_along_axis(
                    remove_time_axis(q_values),
                    add_feature_axis(action),
                    axis=-1,
                )
            )
            _, (next_q_values, _) = self.q_network.apply(
                params,
                *transition.second.to_sequence(),
                initial_carry=next_q_carry,
                rngs={"torso": key},
            )
            next_value = remove_time_axis(next_q_values).max(axis=-1)
            td_error = (
                transition.second.reward
                + self.cfg.gamma * next_value * (1.0 - transition.second.done)
                - q_value
            )
            return q_value, td_error

        (q_values, td_errors), q_vjp = jax.vjp(compute_td_error, state.q_params)

        batch = self.cfg.num_envs
        eye = jnp.eye(batch, dtype=q_values.dtype)
        zeros = jnp.zeros((batch, batch), dtype=q_values.dtype)
        (q_grads,) = jax.vmap(q_vjp)((eye, zeros))
        (td_grads,) = jax.vmap(q_vjp)((zeros, eye))

        def h_fn(params):
            next_carry, (h_values, _) = self.h_network.apply(
                params,
                *transition.first.to_sequence(),
                initial_carry=h_carry,
                rngs={"torso": key},
            )
            h_value = remove_feature_axis(
                jnp.take_along_axis(
                    remove_time_axis(h_values),
                    add_feature_axis(action),
                    axis=-1,
                )
            )
            return h_value, next_carry

        h_values, h_vjp, next_h_carry = jax.vjp(h_fn, state.h_params, has_aux=True)
        (h_grads,) = jax.vmap(h_vjp)(jnp.eye(batch, dtype=h_values.dtype))

        reset_trace = transition.second.done | non_greedy
        discount = jnp.broadcast_to(
            jnp.float32(self.cfg.gamma * self.cfg.trace_lambda), reset_trace.shape
        )
        q_trace_state = self.q_trace.update(
            state.q_trace_state,
            q_grads,
            discount,
            state.q_params,
            partial(
                compute_q_value, timestep=transition.first, carry=q_carry, action=action
            ),
        )
        q_trace_state = self.q_trace.observe(
            q_trace_state,
            transition.first,
            action,
            reset_trace,
            compute_q_value,
            state.q_params,
        )
        h_trace_state = self.h_trace.update(
            state.h_trace_state,
            h_grads,
            discount,
            state.h_params,
            partial(
                compute_h_value, timestep=transition.first, carry=h_carry, action=action
            ),
        )
        h_trace_state = self.h_trace.observe(
            h_trace_state,
            transition.first,
            action,
            reset_trace,
            compute_h_value,
            state.h_params,
        )
        bias_trace_state = self.bias_trace.update(
            state.bias_trace_state,
            h_values,
            discount,
        )

        q_updates = jax.tree.map(
            lambda td_gradient: -broadcast(bias_trace_state.trace, td_gradient)
            * td_gradient,
            td_grads,
        )

        if self.cfg.gradient_correction:
            q_updates = jax.tree.map(
                lambda update, trace, grad: update
                + broadcast(td_errors, trace) * trace
                - broadcast(h_values, grad) * grad,
                q_updates,
                q_trace_state.taylor,
                q_grads,
            )

        h_updates = jax.tree.map(
            lambda trace, grad, param: broadcast(td_errors, trace) * trace
            - broadcast(h_values, grad) * grad
            - self.cfg.regularization_coefficient * param[None],
            h_trace_state.taylor,
            h_grads,
            state.h_params,
        )

        q_grads_final = jax.tree.map(lambda x: -x.mean(axis=0), q_updates)
        h_grads_final = jax.tree.map(lambda x: -x.mean(axis=0), h_updates)

        q_param_updates, q_optimizer_state = self.q_optimizer.update(
            q_grads_final,
            state.q_optimizer_state,
            state.q_params,
        )
        h_param_updates, h_optimizer_state = self.h_optimizer.update(
            h_grads_final,
            state.h_optimizer_state,
            state.h_params,
        )
        q_params = optax.apply_updates(state.q_params, q_param_updates)
        h_params = optax.apply_updates(state.h_params, h_param_updates)

        new_q_trace_state = self.q_trace.reset(q_trace_state, reset_trace)
        new_h_trace_state = self.h_trace.reset(h_trace_state, reset_trace)
        new_bias_trace_state = self.bias_trace.reset(bias_trace_state, reset_trace)

        gamma_lambda = float(self.cfg.gamma * self.cfg.trace_lambda)
        q_target = q_values + td_errors
        explained_variance = 1 - jnp.var(td_errors) / (jnp.var(q_target) + 1e-8)
        log_dict = {
            "q_network/q_value": q_values.mean(),
            "q_network/td_error": td_errors.mean(),
            "q_network/explained_variance": explained_variance,
            "h_network/h_trace": bias_trace_state.trace.mean(),
            "q_network/gradient_norm": optax.global_norm(q_grads_final),
            "q_network/update_norm": optax.global_norm(q_param_updates),
            "h_network/bias_estimate": h_values.mean(),
            "h_network/gradient_norm": optax.global_norm(h_grads_final),
            "h_network/update_norm": optax.global_norm(h_param_updates),
            "training/epsilon": self.epsilon_schedule(state.step),
            "q_trace/trace_norm": optax.global_norm(new_q_trace_state.trace),
            "h_trace/trace_norm": optax.global_norm(new_h_trace_state.trace),
            **self.q_trace.compute_staleness(
                new_q_trace_state,
                q_params,
                compute_q_value,
                gamma_lambda,
                state.update_step,
                label="q_trace",
            ),
            **self.h_trace.compute_staleness(
                new_h_trace_state,
                h_params,
                compute_h_value,
                gamma_lambda,
                state.update_step,
                label="h_trace",
            ),
        }
        if self.cfg.taylor_trace:
            log_dict |= {
                "q_trace/drift_norm": optax.global_norm(new_q_trace_state.drift),
                "q_trace/correction_norm": optax.global_norm(
                    new_q_trace_state.correction
                ),
                "h_trace/drift_norm": optax.global_norm(new_h_trace_state.drift),
                "h_trace/correction_norm": optax.global_norm(
                    new_h_trace_state.correction
                ),
            }
        lox.log(log_dict)

        new_state = dict(
            h_carry=next_h_carry,
            q_params=q_params,
            h_params=h_params,
            q_optimizer_state=q_optimizer_state,
            h_optimizer_state=h_optimizer_state,
            q_trace_state=new_q_trace_state,
            h_trace_state=new_h_trace_state,
            bias_trace_state=new_bias_trace_state,
        )

        return state.replace(**new_state)

    def init(self, key: Key) -> QRCState:
        env_key, q_key, h_key, torso_key = jax.random.split(key, 4)
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
        q_carry = self.q_network.initialize_carry((self.cfg.num_envs, None))
        h_carry = self.h_network.initialize_carry((self.cfg.num_envs, None))
        q_params = self.q_network.init(
            {"params": q_key, "torso": torso_key},
            *timestep.to_sequence(),
            initial_carry=q_carry,
        )
        h_params = self.h_network.init(
            {"params": h_key, "torso": torso_key},
            *timestep.to_sequence(),
            initial_carry=h_carry,
        )
        q_optimizer_state = self.q_optimizer.init(q_params)
        h_optimizer_state = self.h_optimizer.init(h_params)

        q_trace_state = self.q_trace.init(
            q_params,
            self.cfg.num_envs,
            taylor=self.cfg.taylor_trace,
            timestep=timestep,
            carry=q_carry,
            action=action,
        )
        h_trace_state = self.h_trace.init(
            h_params,
            self.cfg.num_envs,
            taylor=self.cfg.taylor_trace,
            timestep=timestep,
            carry=h_carry,
            action=action,
        )
        bias_trace_state = self.bias_trace.init(
            jnp.float32(0.0),
            self.cfg.num_envs,
        )

        state = dict(
            step=0,
            update_step=0,
            timestep=timestep,
            q_carry=q_carry,
            h_carry=h_carry,
            env_state=env_state,
            q_params=q_params,
            h_params=h_params,
            q_optimizer_state=q_optimizer_state,
            h_optimizer_state=h_optimizer_state,
            q_trace_state=q_trace_state,
            h_trace_state=h_trace_state,
            bias_trace_state=bias_trace_state,
        )

        return QRCState(**state)

    def warmup(self, key: Key, state: QRCState, num_steps: int) -> QRCState:
        step_keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._random_action),
            state,
            step_keys,
            unroll=self.cfg.unroll,
        )
        return state

    def train(self, key: Key, state: QRCState, num_steps: int) -> QRCState:
        keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        state, _ = jax.lax.scan(
            partial(self._update_step, policy=self._epsilon_greedy_action),
            state,
            keys,
            unroll=self.cfg.unroll,
        )
        return state

    def evaluate(
        self, key: Key, state: QRCState, num_steps: int, deterministic: bool = True
    ) -> QRCState:
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
            q_carry=self.q_network.initialize_carry((self.cfg.num_envs, None)),
            h_carry=self.h_network.initialize_carry((self.cfg.num_envs, None)),
        )

        state, _ = jax.lax.scan(
            partial(self._step, policy=self._greedy_action),
            state,
            jax.random.split(eval_key, num_steps),
        )
        return state
