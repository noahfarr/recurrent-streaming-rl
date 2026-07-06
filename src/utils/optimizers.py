import lox
import optax


def inject_logger(optimizer_fn, prefix: str = "optimizer"):
    """Wrap an optax optimizer factory so its injected hyperparams are logged.

    Internally calls ``optax.inject_hyperparams`` so callable kwargs like a
    learning-rate schedule become part of the optimizer state. On every
    update, each hyperparam is logged via ``lox.log`` under
    ``f"{prefix}/{hyperparam}"``.
    """
    def factory(**kwargs):
        base = optax.inject_hyperparams(optimizer_fn)(**kwargs)

        def update_fn(updates, state, params=None):
            lox.log({f"{prefix}/{k}": v for k, v in state.hyperparams.items()})
            return base.update(updates, state, params)

        return optax.GradientTransformation(init=base.init, update=update_fn)

    return factory
