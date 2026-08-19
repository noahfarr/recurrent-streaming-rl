from collections.abc import Callable

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant
from flax.typing import Dtype

from src.utils.identity import identity
from src.utils.typing import Array


class ActorCritic(nn.Module):
    actor: Callable
    critic: Callable

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> tuple:
        return self.actor(x, **kwargs), self.critic(x, **kwargs)


class AuxiliaryPrediction(nn.Module):
    head: Callable
    features: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: Array, **kwargs):
        prediction = nn.Dense(
            self.features, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)
        self.sow("auxiliary_losses", "prediction", prediction)
        return self.head(x, **kwargs)


class Projection(nn.Module):
    features: int
    head: Callable = identity
    kernel_init: nn.initializers.Initializer = nn.initializers.orthogonal(jnp.sqrt(2))
    bias_init: nn.initializers.Initializer = nn.initializers.constant(0.0)
    activation_fn: Callable = nn.tanh
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> Array:
        x = nn.Dense(
            self.features,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)
        x = self.activation_fn(x)
        return self.head(x, **kwargs)


class QNetwork(nn.Module):
    action_dim: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> Array:
        q = nn.Dense(
            self.action_dim,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)
        return q.astype(jnp.float32)


class VNetwork(nn.Module):
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> Array:
        v = nn.Dense(
            1,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)
        return v.astype(jnp.float32)


class Categorical(nn.Module):
    action_dim: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> distrax.Categorical:
        logits = nn.Dense(
            self.action_dim,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)
        return distrax.Categorical(logits=logits.astype(jnp.float32))


class Gaussian(nn.Module):
    action_dim: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    state_dependent_std: bool = False
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> distrax.Independent:
        mean = nn.Dense(
            self.action_dim,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x).astype(jnp.float32)
        if self.state_dependent_std:
            scale = jax.nn.softplus(
                nn.Dense(
                    self.action_dim,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )(x).astype(jnp.float32)
            )
        else:
            log_std = self.param("log_std", nn.initializers.zeros, self.action_dim)
            scale = jnp.broadcast_to(jnp.exp(log_std), mean.shape)
        return distrax.Independent(
            distrax.Normal(loc=mean, scale=scale), reinterpreted_batch_ndims=1
        )


class SquashedGaussian(nn.Module):
    action_dim: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    LOG_STD_MIN: float = -10.0
    LOG_STD_MAX: float = 2.0

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> distrax.Transformed:
        temperature = kwargs.get("temperature", 1.0)

        mean = nn.Dense(
            self.action_dim,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x).astype(jnp.float32)
        log_std = nn.Dense(
            self.action_dim,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x).astype(jnp.float32)
        log_std = jnp.clip(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = jnp.exp(log_std) * temperature

        base = distrax.Independent(
            distrax.Normal(loc=mean, scale=std), reinterpreted_batch_ndims=1
        )
        return distrax.Transformed(base, distrax.Block(distrax.Tanh(), ndims=1))


class C51(nn.Module):
    num_atoms: int = 51
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: Array, action: Array, **kwargs) -> Array:
        return nn.Dense(
            self.num_atoms, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(jnp.concatenate([x, action], axis=-1))


class Alpha(nn.Module):
    initial_alpha: float = 1.0

    @nn.compact
    def __call__(self) -> Array:
        return self.param("log_alpha", constant(jnp.log(self.initial_alpha)), ())
