import jax
from functools import partial

@partial(jax.jit, static_argnums=(1,))
def foo(a, val=True):
    print("tracing foo", val)
    return a
@jax.jit
def bar(b):
    print("tracing bar")
    b = foo(b, False)
    return b

foo(1, False)
bar(0)
print('foo cache size', foo._cache_size())
print('bar cache size', bar._cache_size())
foo(1, True)
print('foo cache size', foo._cache_size())
print('bar cache size', bar._cache_size())
foo(1, False)
print('foo cache size', foo._cache_size())
print('bar cache size', bar._cache_size())
