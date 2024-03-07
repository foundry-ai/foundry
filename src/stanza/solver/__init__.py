import jax
import jax.flatten_util

from typing import Callable, Any, Optional, NamedTuple
from stanza.struct import dataclass, field, replace

SolverState = Any
Params = Any

@dataclass(kw_only=True)
class Objective:
    # A custom termination condition
    terminate_condition: Callable[[SolverState], bool] = None
    # A custom per-iteration callback
    step_callback: Callable[[SolverState], SolverState] = None

    # returns a condition which, if == 0,
    # indicates optimality of this objective
    def optimality(self, solver_state : SolverState):
        raise NotImplementedError("optimality() must be implemented")

    # get the elements of the state wrt which we can diff
    def extract_params(self, solver_state : SolverState):
        raise NotImplementedError("extract_params() must be implemented")
    
    def replace_params(self, solver_state : SolverState, params : Params) -> SolverState:
        raise NotImplementedError("replace_params() must be implemented")

class UnsupportedObectiveError(RuntimeError):
    pass

@dataclass
class SolverResult:
    solution : SolverState
    history : Optional[SolverState]

class Solver:
    # Can raise an UnsupportedObjectiveError
    # if the objective is not compatible with this solver
    def run(self, objective, *, history=False, **kwargs) -> SolverResult:
        raise UnsupportedObectiveError("Solver does not support this objective")

# All solver states must have iteration and solved fields
@dataclass
class SolverState:
    iteration: int
    solved: bool

# A solver state for Minimize() objectives
@dataclass
class MinimizeState(SolverState):
    # The function state
    cost: float # the current objective cost
    state: Any
    params: Any
    aux: Any # auxiliary output of the objective

# Fun <= 0
@dataclass
class IneqConstraint:
    fun: Callable

# Fun == 0
@dataclass
class EqConstraint:
    fun: Callable

# Minimize the passed-in function
@dataclass(kw_only=True)
class Minimize(Objective):
    fun: Callable[[Any, Any], float]
    has_state: bool = field(default=False, pytree_node=False)
    has_aux: bool = field(default=False, pytree_node=False)
    initial_state: Any = None # Note that has_state needs to be true in order for this
                           # to be passed into the function!
    initial_params: Any = None

    # Tuple of parameter constraints
    constraints: tuple = ()

    # optimality is when grad wrt params is 0
    def optimality(self, opt_state):
        cost_fn = lambda params: self.eval(opt_state.state, params)[1]
        grad = jax.grad(cost_fn)(opt_state.params)
        return grad
    
    def extract_params(self, opt_state):
        return opt_state.params

    def replace_params(self, opt_state, params):
        return replace(opt_state, params=params)

    # Always of the form (state, params) --> (new_state, cost, aux),
    # and handles the has_state, has_aux cases
    def eval(self, state, params):
        r = self.fun(state, params) if self.has_state else self.fun(params)
        r = (r,) if not (self.has_state or self.has_aux) else r
        state, r = (r[0], r[1:]) if self.has_state else (None, r)
        cost, aux = (r[0], r[1]) if self.has_aux else (r[0], None)
        return state, cost, aux
