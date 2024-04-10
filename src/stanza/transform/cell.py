import jax
import contextlib

from typing import Generic, TypeVar

T = TypeVar('T')

_scope_object = None

@contextlib.contextmanager
def use_scope(scope_object):
    global _scope_object
    _old_scope = _scope_object
    _scope_object = scope_object
    yield
    _scope_object = _old_scope

# Note: not a pytree node because it is mutable
class Cell(Generic[T]):
    def __init__(self, value=None):
        self._value = value
        self._scope = _scope_object

    def get(self):
        assert self._scope == _scope_object, "Cell is not in the current scope!"
        return self._value
    
    def set(self, /, value):
        assert self._scope == _scope_object, "Cell is not in the current scope!"
        self._value = value
    
    def __repr__(self):
        return f"Cell({self._value.__repr__()})"

# Cells are pytree nodes, but note that they get unwrapped as FrozenCell
# so that they cannot be mutated!
jax.tree_util.register_pytree_node(
    Cell,
    lambda x: ((x._value,), None),
    lambda _, children: FrozenCell(*children)
)

# FrozenCell is a valid jax pytree node!
class FrozenCell:
    def __init__(self, value):
        self._value = value
    
    def get(self):
        return self._value
    
    def set(self, /, value):
        raise RuntimeError("Cell is frozen!")

jax.tree_util.register_pytree_node(
    FrozenCell,
    lambda x: ((x._value,), None),
    lambda _, children: FrozenCell(*children)
)

class CellRef:
    def __init__(self, index):
        self.index = index
    
    @staticmethod
    def extract_cells(tree, cells=None):
        # get all of the leaves in the tree that are cells
        leaves, _ = jax.tree_flatten(tree, is_leaf=lambda x: isinstance(x, Cell))
        if not cells:
            new_cells = list({l for l in leaves if isinstance(l, Cell)})
            cells = new_cells
        else:
            cells = list(cells)
            cells_set = set(cells)
            cells.extend({l for l in leaves if isinstance(l, Cell) and l not in cells_set})

        # for any cells we have not seen before,
        # we need to elevate them to top-level cells
        # and keep doing this until all shared cells are top-level
        # this lets us flatten the tree DAG into trees with references to other trees
        visited_cells = set(cells)
        queue = list(cells)
        while queue:
            c = queue.pop()
            leaves = jax.tree_flatten(c._value, is_leaf=lambda x: isinstance(x, Cell))
            new_cells = {l for l in leaves if isinstance(l, Cell) and l not in visited_cells}
            cells.extend(new_cells)
            queue.extend(new_cells)
            visited_cells.update(new_cells)
        return cells

    @staticmethod
    def reference_cells(cells, trees):
        cell_idx = {c: i for i, c in enumerate(cells)}
        def map(c):
            if not isinstance(c, Cell): return c
            return CellRef(cell_idx[c])
        return jax.tree_map(map, trees, is_leaf=lambda x: isinstance(x, Cell))

    @staticmethod
    def resolve_cells(cells, tree):
        return jax.tree_map(
            lambda x: cells[x.index] if isinstance(x, CellRef) else x,
            tree, is_leaf=lambda x: isinstance(x, CellRef)
        )
    
    def __repr__(self):
        return f"CellRef({self.index})"


jax.tree_util.register_pytree_node(
    CellRef,
    lambda x: ((), x.index),
    lambda aux, _: CellRef(aux)
)