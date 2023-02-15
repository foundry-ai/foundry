from jax.tree_util import Partial
import jax

def _sanitize_arg(node):
    if callable(node) and not isinstance(node, Partial):
        node = Partial(node)
    return node

def jit(fun, *args, **kwargs):
    fun = jax.jit(fun, *args, **kwargs)
    def wrapped_fun(*fargs, **fkwargs):
        (fargs, fkwargs) = jax.tree_util.tree_map(_sanitize_arg, (fargs, fkwargs))
        return fun(*fargs, **fkwargs)
    return wrapped_fun