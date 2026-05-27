import functools
import time
from typing import Any, Callable

import jax


def profile(fn: Callable, num_steps: int):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        result = jax.block_until_ready(fn(*args, **kwargs))
        end = time.perf_counter()
        sps = num_steps / (end - start)
        return result, sps

    return wrapper
