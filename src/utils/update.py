import jax
from optax import incremental_update, periodic_update


def periodic_incremental_update(
    new_tensors, old_tensors, steps, update_period, step_size
):
    return periodic_update(
        incremental_update(new_tensors, old_tensors, step_size),
        old_tensors,
        steps,
        update_period,
    )


def delayed_update(new_tensors, old_tensors, steps, start_step):
    return jax.lax.cond(
        steps >= start_step,
        lambda: new_tensors,
        lambda: old_tensors,
    )
