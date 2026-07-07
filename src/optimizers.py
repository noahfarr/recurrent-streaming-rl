import optax
from streamlet.optimizers import ObGD, ObGDConfig, OptaxOptimizer

from src.utils.optimizers import inject_logger


def obgd(name, lr, kappa, beta2=0.999, eps=1e-8, adaptive=False):
    return ObGD(
        cfg=ObGDConfig(lr=lr, kappa=kappa, beta2=beta2, eps=eps, adaptive=adaptive),
        name=name,
    )


def sgd(name, lr, max_grad_norm=None):
    tx = optax.sgd(lr)
    if max_grad_norm is not None:
        tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), tx)
    return OptaxOptimizer(tx=tx, name=name)


def adam(name, lr, eps=1e-5, max_grad_norm=None):
    tx = inject_logger(optax.adam, prefix=name)(learning_rate=lr, eps=eps)
    if max_grad_norm is not None:
        tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), tx)
    return tx


def adamw(name, lr, b2=0.999, eps=1e-8, weight_decay=1e-4, max_grad_norm=None):
    tx = inject_logger(optax.adamw, prefix=name)(
        learning_rate=lr, b2=b2, eps=eps, weight_decay=weight_decay
    )
    if max_grad_norm is not None:
        tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), tx)
    return tx
