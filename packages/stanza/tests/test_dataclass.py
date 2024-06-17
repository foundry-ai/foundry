from stanza.dataclasses import dataclass, field

import jax.numpy as jnp
import jax.tree_util

@dataclass
class A:
    x: int
    y: int
    z: int = field(default=0)

@dataclass(kw_only=True)
class B:
    a: int = None

@dataclass(kw_only=True)
class C(A):
    foo: jax.Array

def test_simple():
    a = A(1,0)
    assert a.x == 1
    assert a.y == 0
    assert a.z == 0

def test_kwonly():
    b = B(a=1)
    assert b.a == 1