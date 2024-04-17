from jax import Array
import jax.numpy as jnp
from jax.core import Shape

import jax.nn.initializers as _init
from functools import wraps
import typing
from typing import Any, Protocol

def _wrap(build):
    @wraps(build)
    def wrapped(*args, **kwargs):
        initializer = build(*args, **kwargs)
        return _wrap_init(initializer)
    return wrapped

def _wrap_init(init):
    @wraps(init)
    def wrapped_initializer(rng, shape, dtype):
        return init(next(rng), shape, dtype)
    return wrapped_initializer


KeyArray = Array
DTypeLikeInexact = Any

@typing.runtime_checkable
class Initializer(Protocol):
  @staticmethod
  def __call__(key: KeyArray,
               shape: Shape,
               dtype: DTypeLikeInexact = jnp.float_) -> Array:
    raise NotImplementedError

constant = _wrap(_init.constant)
delta_orthogonal = _wrap(_init.delta_orthogonal)
glorot_normal = _wrap(_init.glorot_normal)
glorot_uniform = _wrap(_init.glorot_uniform)
he_normal = _wrap(_init.he_normal)
he_uniform = _wrap(_init.he_uniform)
kaiming_normal = _wrap(_init.kaiming_normal)
kaiming_uniform = _wrap(_init.kaiming_uniform)
lecun_normal = _wrap(_init.lecun_normal)
lecun_uniform = _wrap(_init.lecun_uniform)
normal = _wrap(_init.normal)
ones = _wrap(_init.ones)
orthogonal = _wrap(_init.orthogonal)
truncated_normal = _wrap(_init.truncated_normal)
uniform = _wrap(_init.uniform)
variance_scaling = _wrap(_init.variance_scaling)
xavier_normal = _wrap(_init.xavier_normal)
xavier_uniform = _wrap(_init.xavier_uniform)
zeros = _wrap_init(_init.zeros)