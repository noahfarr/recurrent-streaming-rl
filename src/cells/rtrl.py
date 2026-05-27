import jax
import jax.numpy as jnp
import lox
import optax
from flax import linen as nn
from flax import struct
from jax.flatten_util import ravel_pytree
from memorax.networks.sequence_models.rtu import RTUCarry
from memorax.networks.sequence_models.sequence_model import SequenceModel
from memorax.utils import Timestep
from memorax.utils.axes import add_time_axis, get_input_shape, remove_time_axis
from memorax.utils.typing import Array, Carry, PyTree

from src.utils.staleness_buffer import (
    StalenessBuffer,
    StalenessBufferState,
    StalenessStatistics,
)


@struct.dataclass
class RTRLState:
    carry: Carry
    sensitivity: PyTree
    correction: PyTree | None = None
    previous_params: PyTree | None = None
    buffer: StalenessBufferState | None = None
    step: Array = struct.field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))

    @property
    def taylor(self) -> PyTree:
        if self.correction is not None:
            return jax.tree.map(jnp.add, self.sensitivity, self.correction)
        return self.sensitivity


class RTRL(SequenceModel):
    sequence_model: SequenceModel
    track: bool = False
    track_window: int = 32
    staleness_interval: int = 1
    taylor_sensitivity: bool = False
    second_order: bool = False

    @property
    def staleness_buffer(self) -> StalenessBuffer:
        return StalenessBuffer(capacity=self.track_window)

    def compute_staleness(self, state: RTRLState) -> dict[str, Array]:
        if state.buffer is None:
            return {}

        buffer = state.buffer
        _, capacity, *_ = buffer.reset.shape
        chronological = (buffer.index + jnp.arange(capacity)) % capacity

        initial_sensitivity = jax.tree.map(jnp.zeros_like, state.sensitivity)
        initial_carry = buffer.initial_carry

        def step(scan_state, idx):
            carry, sensitivity = scan_state
            obs = buffer.timestep.obs[:, idx]
            reset = buffer.reset[:, idx]
            next_carry, _, sensitivity = self.sequence_model.local_jacobian(
                add_time_axis(obs),
                add_time_axis(reset),
                carry,
                sensitivity=sensitivity,
            )
            return (next_carry, sensitivity), None

        def do_compute(_):
            (_, sensitivity), _ = jax.lax.scan(
                step, (initial_carry, initial_sensitivity), chronological
            )
            is_full = buffer.filled >= capacity
            results = StalenessStatistics.compute(
                state.taylor, sensitivity
            ).to_dict("rtrl")
            return jax.tree.map(lambda x: jnp.where(is_full, x, 0.0), results)

        zero_stats = StalenessStatistics(
            l_1=jnp.float32(0.0),
            max_relative_drift=jnp.float32(0.0),
            l_2=jnp.float32(0.0),
            l_inf=jnp.float32(0.0),
            l1_top_1=jnp.float32(0.0),
            l1_top_5=jnp.float32(0.0),
            l1_top_10=jnp.float32(0.0),
            l1_top_25=jnp.float32(0.0),
            l1_top_50=jnp.float32(0.0),
            cosine_similarity=jnp.float32(0.0),
            cosine_top_1=jnp.float32(0.0),
            cosine_top_5=jnp.float32(0.0),
            cosine_top_10=jnp.float32(0.0),
            cosine_top_25=jnp.float32(0.0),
            cosine_top_50=jnp.float32(0.0),
            participation=jnp.float32(0.0),
        )
        zeros = zero_stats.to_dict("rtrl")

        if self.staleness_interval <= 1:
            results = do_compute(None)
        else:
            results = jax.lax.cond(
                state.step[0] % self.staleness_interval == 0,
                do_compute,
                lambda _: zeros,
                None,
            )
        results = {
            **results,
            "rtrl/sensitivity_norm": optax.global_norm(state.sensitivity),
        }
        if self.taylor_sensitivity:
            results["rtrl/correction_norm"] = optax.global_norm(state.correction)
        return results

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        done: Array,
        initial_carry: Carry | None = None,
        **kwargs,
    ) -> tuple[Carry, Array]:
        if not hasattr(self.sequence_model, "initialize_carry"):
            return None, self.sequence_model(inputs, done, **kwargs)

        if initial_carry is None:
            input_shape = get_input_shape(inputs)
            initial_carry = self.initialize_carry(jax.random.key(0), input_shape)

        state = initial_carry

        if not self.track or self.is_initializing():
            next_carry, y, next_sensitivity = self.sequence_model.local_jacobian(
                inputs, done, state.carry, sensitivity=state.sensitivity, **kwargs
            )
            lox.log(
                {"rtrl/sensitivity_norm": optax.global_norm(next_sensitivity)},
                tags=["training"],
            )
            return state.replace(carry=next_carry, sensitivity=next_sensitivity), y

        carry, sensitivity, correction, previous_params, buffer = (
            state.carry,
            state.sensitivity,
            state.correction,
            state.previous_params,
            state.buffer,
        )

        params = self.variables["params"]["sequence_model"]["cell"]
        features = remove_time_axis(inputs)
        num_envs, *_ = inputs.shape

        if self.taylor_sensitivity:
            delta_params = jax.tree.map(jnp.subtract, params, previous_params)
            sensitivity_input = jax.tree.map(jnp.add, sensitivity, correction)
        else:
            sensitivity_input = sensitivity

        next_carry, y, next_sensitivity = self.sequence_model.local_jacobian(
            inputs, done, carry, sensitivity=sensitivity_input, **kwargs
        )

        if self.taylor_sensitivity:
            def step(params, sensitivity):
                _, _, s = self.sequence_model.apply(
                    {"params": {"cell": params}},
                    inputs,
                    done,
                    carry,
                    sensitivity=sensitivity,
                    method="local_jacobian",
                    **kwargs,
                )
                return s

            _, param_drift = jax.jvp(
                lambda p: step(p, sensitivity),
                (params,),
                (delta_params,),
            )
            _, correction_propagation = jax.jvp(
                lambda s: step(params, s),
                (sensitivity,),
                (correction,),
            )

            if self.second_order:
                # Second-order Taylor in parameters: ½ ∂²step/∂θ² (Δθ, Δθ)
                # via nested JVP-of-JVP-of-step.
                _, param_drift_2 = jax.jvp(
                    lambda p_outer: jax.jvp(
                        lambda p_inner: step(p_inner, sensitivity),
                        (p_outer,),
                        (delta_params,),
                    )[1],
                    (params,),
                    (delta_params,),
                )
                param_drift = jax.tree.map(
                    lambda d, d2: d + 0.5 * d2, param_drift, param_drift_2
                )

            next_sensitivity = jax.tree.map(
                jnp.subtract, next_sensitivity, correction_propagation
            )
            next_correction = jax.tree.map(
                jnp.add, param_drift, correction_propagation
            )
            next_previous_params = params
        else:
            next_correction = None
            next_previous_params = None

        done_flat = remove_time_axis(done)

        def evict(c, ts, a):
            new_carry, _ = self.sequence_model(
                add_time_axis(ts.obs), add_time_axis(ts.done), c,
            )
            return new_carry

        next_buffer = self.staleness_buffer.add(
            buffer,
            Timestep(obs=features, done=done_flat, action=jnp.zeros((num_envs,))),
            jnp.zeros((num_envs,)),
            done_flat,
            evict,
        )

        next_state = state.replace(
            carry=next_carry,
            sensitivity=next_sensitivity,
            correction=next_correction,
            previous_params=next_previous_params,
            buffer=next_buffer,
            step=state.step + 1,
        )
        lox.log(self.compute_staleness(next_state), tags=["training"])
        return next_state, y

    def initialize_carry(self, key, input_shape):
        if not hasattr(self.sequence_model, "initialize_carry"):
            return None

        num_envs, *_ = input_shape

        carry = self.sequence_model.initialize_carry(key, input_shape)
        sensitivity = self.sequence_model.initialize_sensitivity(key, input_shape)

        step = jnp.zeros((num_envs,), dtype=jnp.int32)
        if not self.track:
            return RTRLState(carry=carry, sensitivity=sensitivity, step=step)

        features = self.sequence_model.cell.config.features
        input = jnp.zeros((num_envs, features))

        if self.taylor_sensitivity:
            correction = self.sequence_model.initialize_sensitivity(key, input_shape)
            previous_params = jax.tree.map(
                jnp.zeros_like,
                self.sequence_model.cell.init(key, carry, input)["params"],
            )
        else:
            correction = None
            previous_params = None

        buffer = self.staleness_buffer.init(
            Timestep(
                obs=input,
                done=jnp.zeros((num_envs,), dtype=jnp.bool_),
                action=jnp.zeros((num_envs,)),
            ),
            jnp.zeros((num_envs,)),
            jnp.zeros((num_envs,), dtype=jnp.bool_),
            carry,
        )
        return RTRLState(
            carry=carry,
            sensitivity=sensitivity,
            correction=correction,
            previous_params=previous_params,
            buffer=buffer,
            step=step,
        )
