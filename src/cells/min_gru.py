from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.linen import RNNCellBase
from flax.linen.initializers import lecun_normal, zeros_init

from src.utils.typing import Array

from .injection import custom_jvp
from .rnn import RNN

_PARAM_NAMES = ("W_z", "b_z", "W_h", "b_h")


@struct.dataclass
class MinGRUConfig:
    features: int
    hidden_dim: int
    output_activation_fn: Callable = struct.field(pytree_node=False, default=lambda x: x)


class MinGRUCell(RNNCellBase):
    """Minimal GRU (Feng et al., 2024): the candidate `h_tilde` and gate `z`
    depend only on the input, never on `h_{t-1}`, so the recurrence
    h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t is elementwise/diagonal in h
    exactly (not approximately, as in GRUCell's SnAp1 truncation) — giving an
    exact per-step RTRL Jacobian at O(hidden_dim) cost, the same regime as
    RTUCell, with no dense (H, H) state Jacobian and no need for UORO/SnAp1.
    """

    config: MinGRUConfig
    wrapper = RNN

    @property
    def num_feature_axes(self) -> int:
        return 1

    def setup(self):
        H, F = self.config.hidden_dim, self.config.features
        init = lecun_normal()
        bias_init = zeros_init()
        self.W_z = self.param("W_z", init, (H, F))
        self.b_z = self.param("b_z", bias_init, (H,))
        self.W_h = self.param("W_h", init, (H, F))
        self.b_h = self.param("b_h", bias_init, (H,))

    def _params(self):
        return tuple(getattr(self, name) for name in _PARAM_NAMES)

    @staticmethod
    def _g(x):
        return jnp.where(x >= 0, x + 0.5, nn.sigmoid(x))

    @staticmethod
    def _step(h, W_z, b_z, W_h, b_h, x):
        z = nn.sigmoid(W_z @ x + b_z)
        h_tilde = MinGRUCell._g(W_h @ x + b_h)
        return (1.0 - z) * h + z * h_tilde

    @staticmethod
    def _unit_step(h_i, W_z_i, b_z_i, W_h_i, b_h_i, x):
        z_i = nn.sigmoid(W_z_i @ x + b_z_i)
        h_tilde_i = MinGRUCell._g(W_h_i @ x + b_h_i)
        return (1.0 - z_i) * h_i + z_i * h_tilde_i

    @nn.compact
    def __call__(self, carry: Array, inputs: Array) -> tuple[Array, Array]:
        new_carry = self._step(carry, *self._params(), inputs)
        return new_carry, self.output(new_carry)

    def output(self, carry: Array) -> Array:
        return self.config.output_activation_fn(carry)

    def _flatten_params(self, params):
        return jnp.concatenate(
            [
                params["W_z"],
                params["b_z"][:, None],
                params["W_h"],
                params["b_h"][:, None],
            ],
            axis=-1,
        )

    def local_jacobian(self, carry: Array, inputs: Array, **kwargs):
        params = jax.lax.stop_gradient(self._params())
        h = jax.lax.stop_gradient(carry)
        jacobians = jax.vmap(
            jax.jacfwd(self._unit_step, argnums=(0, 1, 2, 3, 4)),
            in_axes=(0, 0, 0, 0, 0, None),
        )(h, *params, inputs)
        state_jacobian, W_z_jacobian, b_z_jacobian, W_h_jacobian, b_h_jacobian = (
            jacobians
        )
        new_carry = jax.vmap(self._unit_step, in_axes=(0, 0, 0, 0, 0, None))(
            carry, *params, inputs
        )
        parameter_jacobian = jnp.concatenate(
            [
                W_z_jacobian,
                b_z_jacobian[:, None],
                W_h_jacobian,
                b_h_jacobian[:, None],
            ],
            axis=-1,
        )
        return new_carry, state_jacobian, parameter_jacobian

    def propagate_influence(self, state_jacobian, influence):
        return state_jacobian[:, None] * influence

    def inject_influence(self, carry: Array, influence):
        def fn(mdl, h, influence):
            return h

        def jvp_fn(primals, tangents):
            _, h, influence = primals
            variable_tangents, carry_tangent, _ = tangents
            contribution = jnp.einsum(
                "ik,ik->i",
                influence,
                self._flatten_params(variable_tangents["params"]),
            )
            return h, carry_tangent + contribution

        return custom_jvp(fn=fn, jvp_fn=jvp_fn)(self, carry, influence)

    @nn.nowrap
    def initialize_carry(self, key, input_shape):
        return jnp.zeros(self.config.hidden_dim)

    def initialize_influence(self, key, input_shape):
        H, F = self.config.hidden_dim, self.config.features
        return jnp.zeros((H, 2 * F + 2))
