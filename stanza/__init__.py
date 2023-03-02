from functools import partial, wraps
from stanza.util.dataclasses import dataclass
from typing import Callable
import types
import jax

# Tools for auto wrapping and unwrapping
# function arguments
@dataclass(jax=True)
class _FuncWrapper:
    func: Callable

def _wrap_arg(node):
    if callable(node) and not is_jaxtype(type(node)):
        node = _FuncWrapper(node)
    return node

def _unwrap_arg(node):
    if isinstance(node, _FuncWrapper):
        node = node.func
    return node

def _unwrap_is_leaf(node):
    if isinstance(node, _FuncWrapper):
        return True
    return False

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
    def internal_fun(*fargs, **kwargs):
        (fargs, kwargs) = jax.tree_util.tree_map(_unwrap_arg, 
                                                 (fargs, kwargs),
                                                 is_leaf=_unwrap_is_leaf)
        return fun(*fargs, **kwargs)
    jfun = jax.jit(internal_fun, **kwargs)

    @wraps(fun)
    def wrapped_fun(*fargs, **fkwargs):
        (fargs, fkwargs) = jax.tree_util.tree_map(_wrap_arg, (fargs, fkwargs))
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

# returns a version of x
# through which gradients will not flow
# backwards
@jax.custom_vjp
def block_grad(x):
    return x

def _block_grad_fwd(x):
    return x, None

def _block_grad_bkw(_0, _1):
    return None

block_grad.defvjp(_block_grad_fwd, _block_grad_bkw)

# returns a version of fp ("fixed point")
# where the derivatives of fp wrt args come from
# implicitly differentiating
# optimality_fun(fp, args) == 0
def implicit_diff(fun, optimality_fun, fp, args):
    def fun_fwd(fp, args):
        return fp, fp
    def fun_bkw(fp, fp_bar):
        # fp + delta fp = f
        # _, vjpfun = jax.vjp(optimality_fun, fp, args)
        # jax.debug.print("jac: {}", vjpfun(fp_bar))
        return None, None
    fun.defvjp(fun_fwd, fun_bkw)
    return fun(fp, args)