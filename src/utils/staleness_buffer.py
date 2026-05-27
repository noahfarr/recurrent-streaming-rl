from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import struct
from memorax.utils import Timestep
from memorax.utils.typing import Array, Carry


TOP_K_PERCENTS = (1, 5, 10, 25, 50)


@struct.dataclass
class StalenessStatistics:
    l_1: Array
    max_relative_drift: Array
    l_2: Array
    l_inf: Array
    l1_top_1: Array
    l1_top_5: Array
    l1_top_10: Array
    l1_top_25: Array
    l1_top_50: Array
    cosine_similarity: Array
    cosine_top_1: Array
    cosine_top_5: Array
    cosine_top_10: Array
    cosine_top_25: Array
    cosine_top_50: Array
    participation: Array

    @classmethod
    def compute(cls, online: Any, replay: Any) -> "StalenessStatistics":
        def flatten(tree):
            leaves = jax.tree.leaves(tree)
            return jnp.concatenate(
                [leaf.reshape(leaf.shape[0], -1) for leaf in leaves], axis=1
            )

        online = flatten(online)
        replay = flatten(replay)
        difference = online - replay

        abs_replay = jnp.abs(replay)
        abs_difference = jnp.abs(difference)

        reference = jnp.mean(abs_replay, axis=1, keepdims=True) + 1e-12
        relative = abs_difference / reference

        online_norm = jnp.linalg.norm(online, axis=1)
        replay_norm = jnp.linalg.norm(replay, axis=1)
        difference_norm = jnp.linalg.norm(difference, axis=1)
        inner_product = jnp.sum(online * replay, axis=1)
        threshold = 1e-12

        l_2 = jnp.where(
            replay_norm < threshold,
            0.0,
            difference_norm / replay_norm,
        )
        cosine_similarity = jnp.where(
            (online_norm * replay_norm) < threshold,
            1.0,
            inner_product / (online_norm * replay_norm),
        )

        max_replay = jnp.max(abs_replay, axis=1)
        max_difference = jnp.max(abs_difference, axis=1)
        l_inf = jnp.where(
            max_replay < threshold,
            0.0,
            max_difference / max_replay,
        )

        def top_k_mask(percent: int) -> Array:
            cutoff = jnp.quantile(
                abs_replay, 1.0 - percent / 100.0, axis=1, keepdims=True
            )
            return abs_replay >= cutoff

        def l1_top_k(mask: Array) -> Array:
            top_replay = jnp.sum(jnp.where(mask, abs_replay, 0.0), axis=1)
            top_difference = jnp.sum(jnp.where(mask, abs_difference, 0.0), axis=1)
            return jnp.where(
                top_replay < threshold,
                0.0,
                top_difference / top_replay,
            )

        def cosine_top_k(mask: Array) -> Array:
            masked_online = jnp.where(mask, online, 0.0)
            masked_replay = jnp.where(mask, replay, 0.0)
            inner = jnp.sum(masked_online * masked_replay, axis=1)
            on_norm = jnp.linalg.norm(masked_online, axis=1)
            rp_norm = jnp.linalg.norm(masked_replay, axis=1)
            return jnp.where(
                (on_norm * rp_norm) < threshold,
                1.0,
                inner / (on_norm * rp_norm),
            )

        top_values = {}
        for p in TOP_K_PERCENTS:
            mask = top_k_mask(p)
            top_values[f"l1_top_{p}"] = l1_top_k(mask)
            top_values[f"cosine_top_{p}"] = cosine_top_k(mask)

        sum_absolute = jnp.sum(abs_difference, axis=1)
        sum_squared = jnp.sum(abs_difference**2, axis=1)
        _, num_parameters, *_ = abs_difference.shape
        participation = jnp.where(
            sum_squared < threshold,
            1.0,
            (sum_absolute**2) / (num_parameters * sum_squared),
        )

        per_param_relative = abs_difference / (abs_replay + reference)
        return cls(
            l_1=jnp.mean(relative, axis=1),
            max_relative_drift=jnp.max(per_param_relative, axis=1),
            l_2=l_2,
            l_inf=l_inf,
            cosine_similarity=cosine_similarity,
            participation=participation,
            **top_values,
        )

    def to_dict(self, label: str, *, prefix: str = "") -> dict[str, Array]:
        result = {
            f"{label}/{prefix}staleness/l_1": jnp.mean(self.l_1),
            f"{label}/{prefix}staleness/max_relative_drift": jnp.mean(
                self.max_relative_drift
            ),
            f"{label}/{prefix}staleness/l_2": jnp.mean(self.l_2),
            f"{label}/{prefix}staleness/l_inf": jnp.mean(self.l_inf),
            f"{label}/{prefix}cosine_similarity": jnp.mean(self.cosine_similarity),
            f"{label}/{prefix}participation": jnp.mean(self.participation),
        }
        for p in TOP_K_PERCENTS:
            result[f"{label}/{prefix}staleness/l1_top_{p}"] = jnp.mean(
                getattr(self, f"l1_top_{p}")
            )
            result[f"{label}/{prefix}cosine_similarity/top_{p}"] = jnp.mean(
                getattr(self, f"cosine_top_{p}")
            )
        return result


@struct.dataclass
class StalenessBufferState:
    timestep: Timestep
    action: Array
    reset: Array
    initial_carry: Carry
    aux: Array | None = None
    index: Array = struct.field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))
    filled: Array = struct.field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))


@dataclass(frozen=True)
class StalenessBuffer:
    capacity: int

    def init(
        self,
        timestep: Timestep,
        action: Array,
        reset: Array,
        initial_carry: Carry,
        aux: Array | None = None,
    ) -> StalenessBufferState:
        num_environments, *_ = reset.shape
        alloc = lambda leaf: jnp.zeros(
            (num_environments, self.capacity, *leaf.shape[1:]),
            dtype=leaf.dtype,
        )
        return StalenessBufferState(
            timestep=jax.tree.map(alloc, timestep),
            action=alloc(action),
            reset=alloc(reset),
            initial_carry=initial_carry,
            aux=alloc(aux) if aux is not None else None,
            index=jnp.array(0, dtype=jnp.int32),
            filled=jnp.array(0, dtype=jnp.int32),
        )

    def add(
        self,
        state: StalenessBufferState,
        timestep: Timestep,
        action: Array,
        reset: Array,
        evict_fn: Callable[[Carry, Timestep, Array], Carry],
        aux: Array | None = None,
    ) -> StalenessBufferState:
        idx = state.index
        write = lambda b, v: b.at[:, idx].set(v)
        read = lambda b: jax.tree.map(lambda l: l[:, idx], b)
        is_full = state.filled >= self.capacity
        forward_carry = evict_fn(
            state.initial_carry, read(state.timestep), state.action[:, idx]
        )

        return state.replace(
            timestep=jax.tree.map(write, state.timestep, timestep),
            action=write(state.action, action),
            reset=write(state.reset, reset),
            initial_carry=jax.tree.map(
                lambda new, old: jnp.where(is_full, new, old),
                forward_carry,
                state.initial_carry,
            ),
            aux=(
                write(state.aux, aux) if aux is not None else state.aux
            ),
            index=(idx + 1) % self.capacity,
            filled=jnp.minimum(state.filled + 1, self.capacity),
        )
