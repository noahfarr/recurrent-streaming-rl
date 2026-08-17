from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper
from streamlet.utils.typing import Array, EnvParams, Key


@struct.dataclass
class PrecomputedResetState:
    env_state: Any
    observations: Any
    states: Any
    index: Array


class PrecomputedResetWrapper(GymnaxWrapper):
    def __init__(self, env, num_resets: int = 1024):
        assert isinstance(env, environment.Environment), (
            "PrecomputedResetWrapper must wrap a raw gymnax Environment, not "
            f"another wrapper, got {type(env)}"
        )
        super().__init__(env)
        self.num_resets = num_resets

    def reset(
        self, key: Key, params: EnvParams | None = None
    ) -> tuple[Array, PrecomputedResetState]:
        if params is None:
            params = self._env.default_params
        keys = jax.random.split(key, self.num_resets)
        observations, states = jax.vmap(self._env.reset_env, in_axes=(0, None))(
            keys, params
        )
        first = lambda tree: jax.tree.map(lambda leaf: leaf[0], tree)
        return first(observations), PrecomputedResetState(
            env_state=first(states),
            observations=observations,
            states=states,
            index=jnp.int32(0),
        )

    def step(
        self,
        key: Key,
        state: PrecomputedResetState,
        action: int | Array,
        params: EnvParams | None = None,
    ) -> tuple[Array, PrecomputedResetState, Array, Array, dict[str, Any]]:
        if params is None:
            params = self._env.default_params

        obs_step, env_state_step, reward, done, info = self._env.step_env(
            key, state.env_state, action, params
        )

        index = jnp.where(done, (state.index + 1) % self.num_resets, state.index)
        take = lambda tree: jax.tree.map(lambda leaf: leaf[index], tree)
        select = lambda reset_leaf, step_leaf: jax.lax.select(
            done, reset_leaf, step_leaf
        )

        obs = jax.tree.map(select, take(state.observations), obs_step)
        env_state = jax.tree.map(select, take(state.states), env_state_step)

        return (
            obs,
            state.replace(env_state=env_state, index=index),
            reward,
            done,
            info,
        )
