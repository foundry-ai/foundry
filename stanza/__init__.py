import functools
from functools import wraps
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

from jax._src.tree_util import _HashableCallableShim

# A version of partial() which makes arguments static
# into the pytree
# Partial makes them jax-based
class partial(functools.partial):
  def __new__(klass, func, *args, **kw):
    if isinstance(func, functools.partial):
      original_func = func
      func = _HashableCallableShim(original_func)
      out = super().__new__(klass, func, *args, **kw)
      func.func = original_func.func
      func.args = original_func.args
      func.keywords = original_func.keywords
      return out
    else:
      return super().__new__(klass, func, *args, **kw)

jax.tree_util.register_pytree_node(
    partial,
    lambda partial_: ((), (partial_.func, partial_.args, partial_.keywords)),
    lambda func, _: partial(func[0], *func[1], **func[2]),  # type: ignore[index]
)

"""
    A version of jax.jit which
       - Can automatically make function-based arguments
         static by wrapping them using Partial()
"""
def jit(fun=None, **kwargs):
    if fun is None:
        return functools.partial(_jit, **kwargs)
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
from jax._src.api_util import _shaped_abstractify_handlers
from jax._src.tree_util import _registry
from jax._src.core import Tracer, ConcreteArray

# If "t" is a jax type
def is_jaxtype(t):
    if t is str:
        return False
    if t is int or t is float or t is bool:
        return True
    if t in _shaped_abstractify_handlers:
        return True
    if t in _registry:
        return True
    return False

def is_concrete(val):
    if isinstance(val, Tracer) and \
            not isinstance(val.aval, ConcreteArray):
        return False
    return True