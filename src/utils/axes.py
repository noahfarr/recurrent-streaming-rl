from src.utils.typing import Array


def add_time_axis(x):
    return x[:, None]


def broadcast_done(done: Array, leaf: Array) -> Array:
    return done.reshape(done.shape + (1,) * (leaf.ndim - done.ndim))
