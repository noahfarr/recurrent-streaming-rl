from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.linen import RNNCellBase
from flax.linen.initializers import lecun_normal, zeros_init
from jax.experimental import sparse

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
        params = jax.lax.stop_gradient(self._params())
        new_carry = self._step(jax.lax.stop_gradient(carry), *params, inputs)
        jacobians = jax.jacrev(self._step, argnums=tuple(range(10)))(
            carry, *params, inputs
        )
        state_jacobian = jacobians[0]
        parameter_jacobian = dict(zip(_PARAM_NAMES, jacobians[1:10]))
        return new_carry, state_jacobian, parameter_jacobian

    def propagate_influence(self, state_jacobian, influence_leaf):
        return jnp.tensordot(state_jacobian, influence_leaf, axes=1)

    def local_jvp(self, carry: Array, inputs: Array, tangent: Array):
        params = jax.lax.stop_gradient(self._params())
        carry = jax.lax.stop_gradient(carry)
        return jax.jvp(lambda h: self._step(h, *params, inputs), (carry,), (tangent,))

    def local_vjp(self, carry: Array, inputs: Array, cotangent: Array):
        carry = jax.lax.stop_gradient(carry)
        params = jax.lax.stop_gradient(self._params())
        _, vjp_fn = jax.vjp(lambda *p: self._step(carry, *p, inputs), *params)
        return dict(zip(_PARAM_NAMES, vjp_fn(cotangent)))

    def inject_influence(self, carry: Array, influence):
        def fn(mdl, h, influence):
            return h

        def forward_fn(mdl, h, influence):
            return h, influence

        def backward_fn(influence, tangent):
            contract = sparse.sparsify(lambda t, leaf: jnp.tensordot(t, leaf, axes=1))
            g_params = jax.tree.map(
                lambda leaf: contract(tangent, leaf),
                influence,
                is_leaf=lambda leaf: isinstance(leaf, sparse.BCOO),
            )
            return {"params": g_params}, tangent, None

        return nn.custom_vjp(fn=fn, forward_fn=forward_fn, backward_fn=backward_fn)(
            self, carry, influence
        )

    @nn.nowrap
    def initialize_carry(self, key, input_shape):
        return jnp.zeros(self.config.hidden_dim)

    def initialize_influence(self, key, input_shape):
        H, F = self.config.hidden_dim, self.config.features
        return {
            "W_ir": jnp.zeros((H, H, F)),
            "W_hr": jnp.zeros((H, H, H)),
            "b_r": jnp.zeros((H, H)),
            "W_iz": jnp.zeros((H, H, F)),
            "W_hz": jnp.zeros((H, H, H)),
            "b_z": jnp.zeros((H, H)),
            "W_in": jnp.zeros((H, H, F)),
            "W_hn": jnp.zeros((H, H, H)),
            "b_n": jnp.zeros((H, H)),
        }
