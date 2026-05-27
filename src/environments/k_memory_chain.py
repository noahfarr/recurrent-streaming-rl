"""KMemoryChain: every-step k-step recall over a tunable horizon.

At each step the agent observes a fresh random bit and must predict the bit
observed `memory_length` (k) steps ago. Unlike MemoryChain (single recall at
the end of the episode), every step both injects a non-zero immediate Jacobian
into the recurrent state and produces a reward-relevant target, so the RTRL
sensitivity is exercised continuously across the entire episode.

Reward is +1/-1 for each correct/incorrect prediction after a `memory_length`
warmup during which no valid target exists.
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    history: jax.Array  # shape (memory_length + 1,), oldest first
    total_perfect: int
    total_regret: float
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int = 40  # default; overridden via default_params


class KMemoryChain(environment.Environment[EnvState, EnvParams]):
    """At step t, predict the bit observed at step t - memory_length.

    The episode is `memory_length * horizons` steps long, of which the first
    `memory_length` are warmup. With `horizons=8` (default), ~88% of every
    episode is valid recall regardless of how `memory_length` is set.
    """

    def __init__(self, memory_length: int = 5, horizons: int = 8):
        super().__init__()
        self.memory_length = memory_length
        self.horizons = horizons

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=self.memory_length * self.horizons)

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        obs = self.get_obs(state, params)

        target = state.history[0]
        warmup = state.time < self.memory_length
        valid = jnp.logical_not(warmup)
        correct = jnp.logical_and(valid, action == target)
        wrong = jnp.logical_and(valid, action != target)
        reward = jnp.float32(correct) - jnp.float32(wrong)

        new_bit = jax.random.bernoulli(key, p=0.5).astype(jnp.int32)
        new_history = jnp.concatenate(
            [state.history[1:], jnp.asarray([new_bit], dtype=jnp.int32)]
        )

        state = EnvState(
            history=new_history,
            total_perfect=jnp.int32(state.total_perfect + correct),
            total_regret=jnp.float32(state.total_regret + 2 * wrong),
            time=state.time + 1,
        )

        done = self.is_terminal(state, params)
        info = {"discount": self.discount(state, params)}
        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        # Sentinel-zero buffer; reward is masked during the warmup window.
        first_bit = jax.random.bernoulli(key, p=0.5).astype(jnp.int32)
        history = jnp.zeros((self.memory_length + 1,), dtype=jnp.int32)
        history = history.at[-1].set(first_bit)
        state = EnvState(
            history=history,
            total_perfect=jnp.int32(0),
            total_regret=jnp.float32(0),
            time=jnp.int32(0),
        )
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams) -> jax.Array:
        current_bit = state.history[-1]
        bit_signed = 2 * current_bit.astype(jnp.float32) - 1
        time_remaining = 1.0 - state.time / params.max_steps_in_episode
        return jnp.stack([bit_signed, time_remaining])

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        return state.time >= params.max_steps_in_episode

    @property
    def name(self) -> str:
        return "KMemoryChain-v0"

    @property
    def num_actions(self) -> int:
        return 2

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, (2,), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Dict(
            {
                "history": spaces.Box(
                    0, 1, (self.memory_length + 1,), jnp.int32
                ),
                "total_perfect": spaces.Discrete(params.max_steps_in_episode),
                "total_regret": spaces.Discrete(params.max_steps_in_episode),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )
