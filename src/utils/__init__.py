import functools
from typing import Callable, Iterable

import jax
from jax.extend.core import ClosedJaxpr
from lox.stripping import strip_jaxpr
from lox.utils import flatten as _lox_flatten
from lox.utils import is_hashable

from .profile import profile
from .resolvers import *


def strip(
    fun: Callable,
    argnames: Iterable[str] | None = None,
    tags: Iterable[str] | None = None,
) -> Callable:
    """`lox.strip` that treats hashable args as static, mirroring `lox.spool`."""

    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
        args_flat, structure = jax.tree.flatten((args, kwargs))
        static_argnums = tuple(
            i for i, arg in enumerate(args_flat) if is_hashable(arg)
        )
        flat_fn = _lox_flatten(fun, structure)
        closed_jaxpr, out_shape = jax.make_jaxpr(
            flat_fn, static_argnums=static_argnums, return_shape=True
        )(*args_flat)
        new_jaxpr = strip_jaxpr(closed_jaxpr.jaxpr, argnames=argnames, tags=tags)
        closed_jaxpr = ClosedJaxpr(new_jaxpr, closed_jaxpr.consts)
        dynamic_args_flat = tuple(
            arg for arg in args_flat if not is_hashable(arg)
        )
        out_flat = jax.core.eval_jaxpr(
            closed_jaxpr.jaxpr, closed_jaxpr.literals, *dynamic_args_flat
        )
        return jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(out_shape), out_flat
        )

    return wrapped


def flatten(tree):
    from jax.tree_util import DictKey, tree_flatten_with_path
    leaves, _ = tree_flatten_with_path(tree)
    result = {}
    for path, leaf in leaves:
        # Get the keys from the path
        keys = []
        for p in path:
            if isinstance(p, DictKey):
                keys.append(p.key)
            else:
                keys.append(str(p))
        
        # Keep the full hierarchy
        new_key = "/".join(keys)
            
        result[new_key] = leaf
    return result
