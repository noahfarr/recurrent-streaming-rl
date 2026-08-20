import math
from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.linen import RNNCellBase
from flax.linen.initializers import lecun_normal, zeros_init

from src.utils.typing import Array

from .rnn import RNN

_PARAM_NAMES = (
    "W_ir", "W_hr", "b_r",
    "W_iz", "W_hz", "b_z",
    "W_in", "W_hn", "b_n",
)


@struct.dataclass
class GRUConfig:
    features: int
    hidden_dim: int
    output_activation_fn: Callable = struct.field(pytree_node=False, default=lambda x: x)


class GRUCell(RNNCellBase):
    config: GRUConfig
    wrapper = RNN

    @property
    def num_feature_axes(self) -> int:
        return 1

    def setup(self):
        H, F = self.config.hidden_dim, self.config.features
        init = lecun_normal()
        bias_init = zeros_init()
        self.W_ir = self.param("W_ir", init, (H, F))
        self.W_hr = self.param("W_hr", init, (H, H))
        self.b_r = self.param("b_r", bias_init, (H,))
        self.W_iz = self.param("W_iz", init, (H, F))
        self.W_hz = self.param("W_hz", init, (H, H))
        self.b_z = self.param("b_z", bias_init, (H,))
        self.W_in = self.param("W_in", init, (H, F))
        self.W_hn = self.param("W_hn", init, (H, H))
        self.b_n = self.param("b_n", bias_init, (H,))

    def _params(self):
        return tuple(getattr(self, name) for name in _PARAM_NAMES)

    def _param_shapes(self):
        H, F = self.config.hidden_dim, self.config.features
        return (
            (H, F), (H, H), (H,),
            (H, F), (H, H), (H,),
            (H, F), (H, H), (H,),
        )

    def _split_params(self, flat):
        shapes = self._param_shapes()
        offsets = []
        total = 0
        for shape in shapes[:-1]:
            total += math.prod(shape)
            offsets.append(total)
        parts = jnp.split(flat, offsets, axis=-1)
        return {
            name: part.reshape(*flat.shape[:-1], *shape)
            for name, part, shape in zip(_PARAM_NAMES, parts, shapes)
        }

    @property
    def unit_index(self):
        return jnp.concatenate(
            [
                jnp.repeat(jnp.arange(shape[0]), math.prod(shape[1:]))
                for shape in self._param_shapes()
            ]
        )

    @staticmethod
    def _step(h, W_ir, W_hr, b_r, W_iz, W_hz, b_z, W_in, W_hn, b_n, x):
        r = nn.sigmoid(W_ir @ x + W_hr @ h + b_r)
        z = nn.sigmoid(W_iz @ x + W_hz @ h + b_z)
        n = jnp.tanh(W_in @ x + r * (W_hn @ h + b_n))
        return (1.0 - z) * n + z * h

    @nn.compact
    def __call__(self, carry: Array, inputs: Array) -> tuple[Array, Array]:
        new_carry = self._step(carry, *self._params(), inputs)
        return new_carry, self.output(new_carry)

    def output(self, carry: Array) -> Array:
        return self.config.output_activation_fn(carry)

    def local_jacobian(self, carry: Array, inputs: Array, **kwargs):
        H = self.config.hidden_dim
        params = jax.lax.stop_gradient(self._params())
        new_carry = self._step(carry, *params, inputs)
        jacobians = jax.jacrev(self._step, argnums=tuple(range(10)))(
            carry, *params, inputs
        )
        state_jacobian = jacobians[0]
        parameter_jacobian = jnp.concatenate(
            [leaf.reshape(H, -1) for leaf in jacobians[1:10]], axis=-1
        )
        return new_carry, state_jacobian, parameter_jacobian

    def propagate_influence(self, state_jacobian, influence):
        return state_jacobian @ influence

    def local_jvp(self, carry: Array, inputs: Array, tangent: Array):
        params = jax.lax.stop_gradient(self._params())
        carry = jax.lax.stop_gradient(carry)
        return jax.jvp(lambda h: self._step(h, *params, inputs), (carry,), (tangent,))

    def local_vjp(self, carry: Array, inputs: Array, cotangent: Array):
        carry = jax.lax.stop_gradient(carry)
        params = jax.lax.stop_gradient(self._params())
        _, vjp_fn = jax.vjp(lambda *p: self._step(carry, *p, inputs), *params)
        return jnp.concatenate([leaf.reshape(-1) for leaf in vjp_fn(cotangent)])

    def inject_influence(self, carry: Array, influence):
        def fn(mdl, h, influence):
            return h

        def forward_fn(mdl, h, influence):
            return h, influence

        def backward_fn(influence, tangent):
            g_params = self._split_params(tangent @ influence)
            return {"params": g_params}, tangent, None

        return nn.custom_vjp(fn=fn, forward_fn=forward_fn, backward_fn=backward_fn)(
            self, carry, influence
        )

    @staticmethod
    def _unit_step(
        h, h_i, W_ir_i, W_hr_i, b_r_i, W_iz_i, W_hz_i, b_z_i, W_in_i, W_hn_i, b_n_i, x
    ):
        r = nn.sigmoid(W_ir_i @ x + W_hr_i @ h + b_r_i)
        z = nn.sigmoid(W_iz_i @ x + W_hz_i @ h + b_z_i)
        n = jnp.tanh(W_in_i @ x + r * (W_hn_i @ h + b_n_i))
        return (1.0 - z) * n + z * h_i

    def local_jacobian_diagonal(self, carry: Array, inputs: Array):
        params = jax.lax.stop_gradient(self._params())
        h = jax.lax.stop_gradient(carry)
        new_carry = self._step(h, *params, inputs)
        state_jacobian_diagonal = jnp.diagonal(
            jax.jacrev(self._step, argnums=0)(h, *params, inputs)
        )
        unit_jacobians = jax.vmap(
            jax.jacfwd(self._unit_step, argnums=tuple(range(2, 11))),
            in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None),
        )(h, h, *params, inputs)
        parameter_jacobian_diagonal = jnp.concatenate(
            [leaf.reshape(-1) for leaf in unit_jacobians]
        )
        return new_carry, state_jacobian_diagonal, parameter_jacobian_diagonal

    def inject_influence_rank1(self, carry: Array, u, v):
        def fn(mdl, h, u, v):
            return h

        def forward_fn(mdl, h, u, v):
            return h, (u, v)

        def backward_fn(residual, tangent):
            u, v = residual
            g_params = self._split_params((tangent @ u) * v)
            return {"params": g_params}, tangent, None, None

        return nn.custom_vjp(fn=fn, forward_fn=forward_fn, backward_fn=backward_fn)(
            self, carry, u, v
        )

    def inject_influence_diagonal(self, carry: Array, influence):
        unit_index = self.unit_index

        def fn(mdl, h, influence):
            return h

        def forward_fn(mdl, h, influence):
            return h, influence

        def backward_fn(influence, tangent):
            g_params = self._split_params(tangent[unit_index] * influence)
            return {"params": g_params}, tangent, None

        return nn.custom_vjp(fn=fn, forward_fn=forward_fn, backward_fn=backward_fn)(
            self, carry, influence
        )

    @nn.nowrap
    def initialize_carry(self, key, input_shape):
        return jnp.zeros(self.config.hidden_dim)

    def initialize_influence(self, key, input_shape):
        H, F = self.config.hidden_dim, self.config.features
        return jnp.zeros((H, 3 * (H * F + H * H + H)))
