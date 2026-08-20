from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.typing import Dtype
from jax.nn.initializers import constant, lecun_normal

from flax.linen import RNNCellBase

from src.utils.typing import Array

from .rnn import RNN


@struct.dataclass
class GTUConfig:
    features: int
    hidden_dim: int
    r_min: float = 0.0
    r_max: float = 1.0
    max_phase: float = 6.28
    gate_bias: float = 2.0
    eps: float = 1e-8
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    activation_fn: Callable = struct.field(pytree_node=False, default=jnp.tanh)
    output_activation_fn: Callable = struct.field(
        pytree_node=False, default=lambda x: x
    )


@struct.dataclass
class GTUCarry:
    real: Array
    imaginary: Array


class GTUCell(RNNCellBase):
    config: GTUConfig
    wrapper = RNN

    @property
    def num_feature_axes(self) -> int:
        return 1

    @staticmethod
    def _init_nu(key, shape, r_min=0.0, r_max=1.0):
        u = jax.random.uniform(key, shape=shape)
        return jnp.log(-0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2))

    @staticmethod
    def _init_theta(key, shape, max_phase=6.28):
        u = jax.random.uniform(key, shape=shape)
        return jnp.log(max_phase * u)

    def setup(self):
        self.nu_log = self.param(
            "nu_log",
            partial(self._init_nu, r_min=self.config.r_min, r_max=self.config.r_max),
            (self.config.hidden_dim,),
        )
        self.theta_log = self.param(
            "theta_log",
            partial(self._init_theta, max_phase=self.config.max_phase),
            (self.config.hidden_dim,),
        )
        self.gate_bias = self.param(
            "gate_bias", constant(self.config.gate_bias), (self.config.hidden_dim,)
        )
        self.gate_kernel = self.param(
            "gate_kernel",
            lecun_normal(),
            (self.config.hidden_dim, self.config.features),
        )
        self.B_real = self.param(
            "B_real", lecun_normal(), (self.config.hidden_dim, self.config.features)
        )
        self.B_imag = self.param(
            "B_imag", lecun_normal(), (self.config.hidden_dim, self.config.features)
        )

    def _unit_step(self, h, nu, th, gb, gk, br, bi, x):
        real, imag = h
        gate = jax.nn.sigmoid(x @ gk + gb)
        r = jnp.exp(-jnp.exp(nu) * gate)
        theta = jnp.exp(th) * gate
        g, phi = r * jnp.cos(theta), r * jnp.sin(theta)
        norm = jnp.sqrt(1 - r**2) + self.config.eps
        pre = jnp.stack(
            [
                g * real - phi * imag + norm * (x @ br),
                g * imag + phi * real + norm * (x @ bi),
            ]
        )
        return self.config.activation_fn(pre)

    @nn.compact
    def __call__(self, carry: GTUCarry, inputs: Array) -> tuple[GTUCarry, Array]:
        h = jnp.stack([carry.real, carry.imaginary], axis=-1)
        new_h = jax.vmap(self._unit_step, in_axes=(0, 0, 0, 0, 0, 0, 0, None))(
            h,
            self.nu_log,
            self.theta_log,
            self.gate_bias,
            self.gate_kernel,
            self.B_real,
            self.B_imag,
            inputs,
        )
        new_carry = GTUCarry(real=new_h[:, 0], imaginary=new_h[:, 1])
        return new_carry, self.output(new_carry)

    def output(self, carry):
        return self.config.output_activation_fn(
            jnp.concatenate([carry.real, carry.imaginary])
        )

    def local_jacobian(self, carry, inputs, **kwargs):
        carry = jnp.stack([carry.real, carry.imaginary], axis=-1)
        params = (
            self.nu_log,
            self.theta_log,
            self.gate_bias,
            self.gate_kernel,
            self.B_real,
            self.B_imag,
        )

        new_carry = jax.vmap(self._unit_step, in_axes=(0, 0, 0, 0, 0, 0, 0, None))(
            carry,
            *jax.lax.stop_gradient(params),
            inputs,
        )
        jacobians = jax.vmap(
            jax.jacfwd(self._unit_step, argnums=(0, 1, 2, 3, 4, 5, 6)),
            in_axes=(0, 0, 0, 0, 0, 0, 0, None),
        )(carry, *params, inputs)
        (
            state_jacobian,
            nu_jacobian,
            theta_jacobian,
            gate_bias_jacobian,
            gate_kernel_jacobian,
            b_real_jacobian,
            b_imag_jacobian,
        ) = jacobians

        parameter_jacobian = jnp.moveaxis(
            jnp.concatenate(
                [
                    nu_jacobian[..., None],
                    theta_jacobian[..., None],
                    gate_bias_jacobian[..., None],
                    gate_kernel_jacobian,
                    b_real_jacobian,
                    b_imag_jacobian,
                ],
                axis=-1,
            ),
            1,
            0,
        )

        carry = GTUCarry(real=new_carry[:, 0], imaginary=new_carry[:, 1])
        return carry, state_jacobian, parameter_jacobian

    def propagate_influence(self, state_jacobian, influence):
        return jnp.einsum("icd,dik->cik", state_jacobian, influence)

    def _split_params(self, flat):
        F = self.config.features
        return {
            "nu_log": flat[..., 0],
            "theta_log": flat[..., 1],
            "gate_bias": flat[..., 2],
            "gate_kernel": flat[..., 3 : 3 + F],
            "B_real": flat[..., 3 + F : 3 + 2 * F],
            "B_imag": flat[..., 3 + 2 * F : 3 + 3 * F],
        }

    def inject_influence(self, carry, influence):

        def fn(mdl, r, i, influence):
            return r, i

        def forward_fn(mdl, real, imag, influence):
            return (real, imag), influence

        def backward_fn(influence, tangents):
            g_real, g_imag = tangents
            tangent = jnp.stack([g_real, g_imag])
            g_params = self._split_params(
                jnp.einsum("ci,cik->ik", tangent, influence)
            )
            return {"params": g_params}, g_real, g_imag, None

        real, imag = nn.custom_vjp(
            fn=fn,
            forward_fn=forward_fn,
            backward_fn=backward_fn,
        )(self, carry.real, carry.imaginary, influence)
        return GTUCarry(real=real, imaginary=imag)

    @nn.nowrap
    def initialize_carry(self, key, input_shape):
        zeros = jnp.zeros(self.config.hidden_dim)
        return GTUCarry(real=zeros, imaginary=zeros)

    def initialize_influence(self, key, input_shape):
        H, F = self.config.hidden_dim, self.config.features
        return jnp.zeros((2, H, 3 * F + 3))
