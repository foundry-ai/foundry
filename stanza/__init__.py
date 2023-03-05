from functools import partial, wraps
from jax.tree_util import Partial
import jax.tree_util
from typing import Callable
import types
import jax

# Tools for auto wrapping and unwrapping
# function arguments
class _FuncWrapper:
    def __init__(self, func):
        self.func = func

def _wrapper_flatten(v):
    return (), v.func

def _wrapper_unflatten(aux, children):
    return _FuncWrapper(aux)

jax.tree_util.register_pytree_node(_FuncWrapper, _wrapper_flatten, _wrapper_unflatten)

def _wrap(args):
    def _wrap_arg(node):
        if callable(node) and not is_jaxtype(type(node)):
            node = _FuncWrapper(node)
        return node
    return jax.tree_util.tree_map(_wrap_arg, args)

def _unwrap(args):
    def _unwrap_arg(node):
        if isinstance(node, _FuncWrapper):
            node = node.func
        return node

    def _unwrap_is_leaf(node):
        if isinstance(node, _FuncWrapper):
            return True
        return False
    return jax.tree_util.tree_map(_unwrap_arg, args, is_leaf=_unwrap_is_leaf)


"""
    A version of jax.jit which
       - Can automatically make function-based arguments
         static by wrapping them using Partial()
"""
def jit(fun=None, **kwargs):
    if fun is None:
        return partial(_jit, **kwargs)
    return _jit(fun, **kwargs)

def _jit(fun, **kwargs):
    @wraps(fun)
    def internal_fun(*fargs, **fkwargs):
        (fargs, fkwargs) = _unwrap((fargs, fkwargs))
        return fun(*fargs, **fkwargs)
    jfun = jax.jit(internal_fun, **kwargs)

    @wraps(fun)
    def wrapped_fun(*fargs, **fkwargs):
        (fargs, fkwargs) = _wrap((fargs, fkwargs))
        return jfun(*fargs, **fkwargs)
    return wrapped_fun


# TODO: Maybe move some of the functions below into separate utils?

# If "t" is a jax type
def is_jaxtype(t):
    from jax._src.api_util import _shaped_abstractify_handlers
    from jax._src.tree_util import _registry
    if t is str:
        return False
    if t is int or t is float or t is bool:
        return True
    if t in _shaped_abstractify_handlers:
        return True
    if t in _registry:
        return True
    return False