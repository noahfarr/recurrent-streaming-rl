from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper
from streamlet.utils.typing import Array, EnvParams, Key


@struct.dataclass
class NextStepAutoResetState:
    env_state: Any
    needs_reset: Array


class NextStepAutoResetWrapper(GymnaxWrapper):

    def __init__(self, env):
        assert isinstance(env, environment.Environment), (
            "NextStepAutoResetWrapper must wrap a raw gymnax Environment, not "
            f"another wrapper, got {type(env)}"
        )
        super().__init__(env)

    def reset(
        self, key: Key, params: EnvParams | None = None
    ) -> tuple[Array, NextStepAutoResetState]:
        obs, env_state = self._env.reset(key, params)
        return obs, NextStepAutoResetState(env_state, jnp.bool_(False))

    def step(
        self,
        key: Key,
        state: NextStepAutoResetState,
        action: int | Array,
        params: EnvParams | None = None,
    ) -> tuple[Array, NextStepAutoResetState, Array, Array, dict[str, Any]]:
        if params is None:
            params = self._env.default_params
        key_step, key_reset = jax.random.split(key)

        obs_step, env_state_step, reward, done, info = self._env.step_env(
            key_step, state.env_state, action, params
        )
        obs_reset, env_state_reset = self._env.reset_env(key_reset, params)

        max_steps_in_episode = getattr(
            params, "max_steps_in_episode", getattr(self._env, "max_steps_in_episode", None)
        )
        if max_steps_in_episode is None:
            truncated = jnp.zeros_like(done)
        else:
            truncated = done & (env_state_step.time >= max_steps_in_episode)
        terminated = done & ~truncated

        def select(reset_leaf, step_leaf):
            return jax.lax.select(state.needs_reset, reset_leaf, step_leaf)

        obs = select(obs_reset, obs_step)
        env_state = jax.tree.map(select, env_state_reset, env_state_step)
        reward = select(jnp.zeros_like(reward), reward)
        done = select(jnp.zeros_like(done), done)
        terminated = select(jnp.zeros_like(terminated), terminated)
        truncated = select(jnp.zeros_like(truncated), truncated)
        info = jax.tree.map(lambda leaf: select(jnp.zeros_like(leaf), leaf), info)
        info = {**info, "terminated": terminated, "truncated": truncated}

        return obs, NextStepAutoResetState(env_state, done), reward, done, info
