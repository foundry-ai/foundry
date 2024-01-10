import pytest

from stanza.struct import struct
from stanza.struct.frozen import FrozenInstanceError

@struct
class Foo:
    a: int
    b: int = 3

def test_simple():
    foo = Foo(1,2)
    assert foo.a == 1
    assert foo.b == 2
    foo = Foo(4)
    assert foo.a == 4
    assert foo.b == 3

@struct(frozen=True)
class FrozenFoo:
    a: int
    b: int

def test_frozen():
    f = FrozenFoo(1,2)
    with pytest.raises(FrozenInstanceError):
        f.a = 2
    with pytest.raises(FrozenInstanceError):
        f.b = 1