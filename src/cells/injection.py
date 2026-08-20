import jax
import jax.numpy as jnp
from flax.core.lift import pack
from flax.linen.transforms import decorator_lift_transform


def _custom_jvp_single_scope_fn(fn, jvp_fn, grad_vars="params", nondiff_argnums=()):
    def inner(scope_fn, repack_fn, variable_groups, rng_groups, *args):
        grad_variables, other_variables = variable_groups

        def f(grad_variables, *args):
            scope = scope_fn((grad_variables, other_variables), rng_groups)
            y = fn(scope, *args)
            return y, repack_fn(scope)

        f = jax.custom_jvp(f, nondiff_argnums=nondiff_argnums)

        def f_jvp(primals, tangents):
            grad_variables, *args = primals
            grad_tangents, *arg_tangents = tangents
            scope = scope_fn((grad_variables, other_variables), rng_groups)
            y, y_dot = jvp_fn(
                (grad_variables[0], *args), (grad_tangents[0], *arg_tangents)
            )
            variables_out = repack_fn(scope)
            return (y, variables_out), (
                y_dot,
                jax.tree.map(jnp.zeros_like, variables_out),
            )

        f.defjvp(f_jvp)

        return f(grad_variables, *args)

    return pack(
        inner,
        (grad_vars, True),
        (grad_vars, True),
        (True,),
        name="custom_jvp",
    )


def custom_jvp(fn, jvp_fn, grad_vars="params", nondiff_argnums=()):
    return decorator_lift_transform(
        _custom_jvp_single_scope_fn,
        fn,
        jvp_fn=jvp_fn,
        grad_vars=grad_vars,
        nondiff_argnums=nondiff_argnums,
        multi_scope=False,
    )
