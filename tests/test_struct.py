import stanza.struct as struct

@struct.dataclass
class A:
    x: int
    y: int = 0
    z: int = struct.field(default=0)

@struct.dataclass(kw_only=True)
class B:
    a: int = None

def test_simple():
    a = A(1)
    assert a.x == 1
    assert a.y == 0
    assert a.z == 0

def test_kwonly():
    b = B(a=1)
    assert b.a == 1