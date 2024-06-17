import jax.random
import jax.numpy

from stanza import struct
from stanza.transform.cell import Cell, FrozenCell

def key_or_seed(key_or_seed):
    if isinstance(key_or_seed, int):
        key_or_seed = jax.random.PRNGKey(key_or_seed)
    elif (hasattr(key_or_seed, "shape") and
            hasattr(key_or_seed, "dtype") and \
            key_or_seed.dtype == jax.numpy.uint32):
        key_or_seed = key_or_seed
    else:
        raise ValueError("Not key or seed!")
    return key_or_seed

@jax.tree_util.register_pytree_node_class
class PRNGSequence:
    def __init__(self, key_or_val, _cell=None):
        if _cell is not None:
            self._cell = _cell
        else:
            self._cell = Cell(key_or_seed(key_or_val))
    
    def __next__(self):
        k, n = jax.jit(
            jax.random.split,
            static_argnums=1
        )(self._cell.get())
        self._cell.set(k)
        return n
    
    def tree_flatten(self):
        return ((self._cell,), None)

    def __repr__(self):
        return f"PRNGSequence({self._cell.__repr__()})"
    
    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(None, *children)