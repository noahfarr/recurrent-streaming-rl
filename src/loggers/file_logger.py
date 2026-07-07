from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

from streamlet.utils.typing import PyTree


class FileLogger:
    """Persists every `log()` call's per-epoch metrics to a single .npz file.

    Mirrors DashboardLogger's reduction: each leaf is `(num_seeds, T, *rest)`,
    reduced with a NaN-aware mean over every axis after the seed axis to a
    `(num_seeds,)` per-epoch value. Values across calls are stacked into
    `(num_epochs, num_seeds)` arrays and written out in `finish()`, so the
    full training curve survives even when stdout only shows a live dashboard.
    """

    def __init__(self, path: str = "metrics.npz", **kwargs):
        self.path = path
        self.history = defaultdict(list)
        self.steps = []

    @staticmethod
    def _epoch_mean(leaf: PyTree):
        leaf = jnp.asarray(leaf)
        finite = jnp.isfinite(leaf)
        if not bool(jnp.any(finite)):
            return None
        axes = tuple(range(1, leaf.ndim))
        total = jnp.nansum(leaf, axis=axes)
        count = jnp.sum(finite, axis=axes)
        return np.asarray(total / jnp.maximum(count, 1))

    def log(self, data: PyTree, steps: PyTree, **kwargs) -> None:
        step = int(jnp.asarray(steps).reshape(-1)[-1])
        self.steps.append(step)
        for path, leaf in jax.tree_util.tree_leaves_with_path(data):
            key = "/".join(str(p.key) for p in path)
            mean = self._epoch_mean(leaf)
            if mean is not None:
                self.history[key].append(mean)

    def log_artifact(self, state: PyTree, step: int, **kwargs) -> None:
        pass

    def finish(self) -> None:
        arrays = {k: np.stack(v, axis=0) for k, v in self.history.items()}
        np.savez(self.path, steps=np.asarray(self.steps), **arrays)
