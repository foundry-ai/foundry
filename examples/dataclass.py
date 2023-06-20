import stanza
from stanza.dataclasses import dataclass, field
import jax.numpy as jnp
from functools import partial
from typing import Callable

# jax=True registers the type and enables frozen=True
@dataclass(jax=True)
class Foo:
    f : int = field(jax_static=True)
    g : float = 0
    h : Callable = None

# You can query whether things are jax types
print("int: ", stanza.is_jaxtype(type(1)))
print("array: ", stanza.is_jaxtype(type(jnp.array([0]))))
print("str: ", stanza.is_jaxtype(type("foo")))
print("func: ", stanza.is_jaxtype(type(lambda x: x)))
print("partial: ", stanza.is_jaxtype(type(partial(lambda x: x))))
print("dataclass: ", stanza.is_jaxtype(Foo))

@stanza.jit
def bar(foo):
    # Notice that foo.f is always concretized!
    print('Tracing bar: ', foo.f)
    return foo.f + foo.g

assert bar(Foo(1, 0, bar)) == 1
assert bar(Foo(1, 1)) == 2
assert bar(Foo(0, 0)) == 0
assert bar(Foo(0, 1)) == 1

def g(x):
    return x + 1

# # stanza.jit allows for function arguments
# # to be automatically treated as static!
# # Additionally it can directly be used as a decorator
# # without having to do partial() all the time
# @stanza.jit(static_argnums=(2,))
# def f(x, y, h):
#     print('Tracing f: ', g)
#     return g(x) + y
# foo = Foo(1, 0, g)

# # Note that "Tracing" only prints once!
# assert f(g, 0, 0, foo) == 1
# assert f(g, 1, 0, None) == 2