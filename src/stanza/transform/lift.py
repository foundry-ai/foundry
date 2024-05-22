import jax
import jax.numpy as jnp
import weakref

from . import is_array
from .cell import Cell, FrozenCell, CellRef, use_scope

from typing import Any, NamedTuple, TypeVar
from typing_extensions import Self
from functools import partial

T = TypeVar('T')

class Static:
    def __init__(self, value):
        self.value = value
    
    @staticmethod
    def static_wrap(tree):
        return jax.tree_util.tree_map(
            lambda x: Static(x) if not is_array(x) else x,
            tree
        )
    
    @staticmethod
    def static_unwrap(tree):
        return jax.tree_util.tree_map(
            lambda x: x.value if isinstance(x, Static) else x,
            tree, is_leaf=lambda x: isinstance(x, Static)
        )

jax.tree_util.register_pytree_node(
    Static,
    lambda x: ((), x.value),
    lambda aux, _: Static(aux)
)

class LiftedFunction:
    def __init__(self, function):
        self.__func__ = function

    def __get__(self, obj, objtype=None):
        return partial(self, obj)

    def __call__(self, cell_states, /, *args, **kwargs):
        # create new cells to pass to the lifted function
        scope = object()
        with use_scope(scope):
            cells = [Cell() for _ in cell_states]
            # inflate the cell states into the cells (they can have cyclic references!)
            cell_states = Static.static_unwrap(cell_states)
            cell_states = CellRef.resolve_cells(cells, cell_states)
            for c, s in zip(cells, cell_states):
                c._value = s
            # un-staticify the arguments if any are wrapped with Static
            # and replace any cells in the arguments themselves
            args, kwargs = Static.static_unwrap((args, kwargs))
            args, kwargs = CellRef.resolve_cells(cells, (args, kwargs))
            # call the function
            res = self.__func__(*args, **kwargs)
            # extract any new cells from the result,
            # may be more than the ones we passed in,
            # but include cells that were passed in
            ret_cells = CellRef.extract_cells(res, cells)
            res = CellRef.reference_cells(ret_cells, res)

            cell_states = [c._value for c in ret_cells]
            cell_states = CellRef.reference_cells(ret_cells, cell_states)
        return cell_states, res

class LoweredFunction:
    def __init__(self, lifted_function):
        self.__func__ = lifted_function

    def __get__(self, obj, objtype=None):
        return partial(self, obj)

    def __call__(self, *args, **kwargs):
        # extract the mutables in the argument
        cells = CellRef.extract_cells((args, kwargs))
        cell_states = [c._value for c in cells]
        cell_states, args, kwargs = CellRef.reference_cells(cells, (cell_states, args, kwargs))
        
        cell_states, args, kwargs = Static.static_wrap((cell_states, args, kwargs))
        # call the lifted function with the jaxified state, arguments
        ret_cell_states, res = self.__func__(cell_states, *args, **kwargs)
        # create new cells for returned cell states
        # that are not associated with any cells
        ret_cells = cells + [Cell() for _ in range(len(ret_cell_states) - len(cell_states))]
        # resolve sub-cells in the returned cell states
        ret_cell_states = CellRef.resolve_cells(ret_cells, ret_cell_states)
        for c, s in zip(ret_cells, ret_cell_states):
            c._value = s
        # resolve sub-cells in the result
        res = CellRef.resolve_cells(ret_cells, res)
        return res

def static_lower(function):
    def lowered(*args, **kwargs):
        args, kwargs = Static.static_unwrap((args, kwargs))
        return function(*args, **kwargs)
    return lowered

_LIFT_CACHE = weakref.WeakKeyDictionary()
_LOWER_CACHE = weakref.WeakKeyDictionary()

def lower(lifted_function):
    return _LOWER_CACHE.setdefault(lifted_function, LoweredFunction(lifted_function))

def lift(function):
    return _LIFT_CACHE.setdefault(function, LiftedFunction(function))