import jax
import jax.flatten_util
import jax.numpy as jnp

import optax

from typing import NamedTuple, Any, Callable
from functools import partial
from stanza.util.dataclasses import dataclass

class SolverResults(NamedTuple):
    solved: bool
    final_params: Any
    final_state: Any
    history: Any

class IterativeSolver:
    # Must have 'max_iterations' attribute
    #
    # The solver state must have a "solved" boolean

    # To be overridden by the iterative solver
    def init_state(self, params, *args, **kwargs):
        raise NotImplementedError("Must be implemented")

    def update(self, params, state, *args, **kwargs):
        raise NotImplementedError("Must be implemented")

    def _do_step(self, state):
        (params, opt_state, i), (args, kwargs) = state
        new_params, new_state = self.update(params, opt_state, *args, **kwargs)
        return (new_params, new_state, i + 1), (args, kwargs)

    def _step_with_history(self, state, arg=None):
        (_, opt_state, _), (_, _) = state
        # Only do a step if we are not done yet
        new_state = jax.lax.cond(opt_state.solved,
            lambda s: s,
            self._do_step,  state)
        return new_state, opt_state

    def _loop_no_history(self, init_params, args, kwargs):
        init_opt_state = self.init_state(init_params, *args, **kwargs)
        loop_state = (init_params, init_opt_state, 0), (args, kwargs)
        loop_state = jax.lax.while_loop(
                lambda s: jnp.logical_and(jnp.logical_not(s[0][1].solved), s[0][2] < self.max_iterations),
                self._do_step, loop_state)
        (final_params, final_state, _), _ = loop_state
        return final_params, final_state, None

    # Suitable for wrapping with implicit diffs
    def _scan_with_history(self, init_params, args, kwargs):
        init_opt_state = self.init_state(init_params, *args, **kwargs)
        scan_state = (init_params, init_opt_state, 0), (args, kwargs)

        # if we want the solver history,
        # use a scan
        scan_state, state_history = \
            jax.lax.scan(self._step_with_history, scan_state,
                        None, length=self.max_iterations)
        (final_params, final_state, _), _ = scan_state
        return final_params, final_state, state_history

    def _run_optimality_fun(self, params_state, args, kwargs):
        params, state = params_state
        return self.optimality_fun(params, state, *args, **kwargs)

    def run(self, init_params, *args, history=False, **kwargs):
        run = self._scan_with_history if history else self._loop_no_history
        final_params, final_state, history = run(init_params, args, kwargs)
        return SolverResults(solved=final_state.solved,
            final_params=final_params,
            final_state=final_state,
            history=history
        )

class OptaxState(NamedTuple):
    cost: float
    aux: Any
    opt_state: Any
    solved: bool

@dataclass
class OptaxSolver(IterativeSolver):
    fun: Callable
    optax_transform: Any
    terminate: Callable = None
    has_aux: bool = False
    tol: float = 1e-3
    max_iterations: int = 50

    def _cost_fun(self, params, *args, **kwargs):
        cost, aux = (self.fun(params, *args, **kwargs)
            if self.has_aux else
            (self.fun(params, *args, **kwargs), None)
        )
        return cost, aux

    def init_state(self, params, *args, **kwargs):
        cost, aux = self._cost_fun(params, *args, **kwargs)
        opt_state = self.optax_transform.init(params)
        return OptaxState(cost, aux, opt_state, False)

    def update(self, params, state, *args, **kwargs):
        fun = lambda v: self._cost_fun(v, *args, **kwargs)[0]
        grad = jax.grad(fun)(params)
        updates, opt_state = self.optax_transform.update(grad, state.opt_state)
        new_params = optax.apply_updates(params, updates)

        cost, aux = self._cost_fun(new_params, *args, **kwargs)

        # check the distance for solution
        if self.terminate:
            solved = self.terminate(cost, aux)
        else:
            solved = False

        return new_params, OptaxState(cost, aux, opt_state, solved)

class NewtonState(NamedTuple):
    cost: float
    aux: Any
    solved: bool

# A newton solver with backtracking support
@dataclass
class NewtonSolver(IterativeSolver):
    fun: Callable
    terminate: Callable = None
    has_aux: bool = False
    tol: float = 1e-2
    max_iterations: int = 500
    # backtracking beta
    beta: float = 0.5

    # Will handle both the has_aux and not-has_aux cases
    def _cost_fun(self, params, *args, **kwargs):
        cost, aux = (self.fun(params, *args, **kwargs)
            if self.has_aux else
            (self.fun(params, *args, **kwargs), None)
        )
        cost = jnp.nan_to_num(cost, nan=float('inf'))
        return cost, aux

    # when gradient is zero we are at the optimal
    def optimality_fun(self, params, state, *args, **kwargs):
        vec_fun = lambda v: self._cost_fun(v, *args, **kwargs)[0]
        grad = jax.grad(vec_fun)(params)
        return grad

    def init_state(self, params, *args, **kwargs):
        cost, aux = self._cost_fun(params, *args, **kwargs)
        return NewtonState(cost, aux, False)
    
    def update(self, params, state, *args, **kwargs):
        # unravel argument structure
        param_v, p_fmt = jax.flatten_util.ravel_pytree(params)
        vec_fun = lambda v: self._cost_fun(p_fmt(v), *args, **kwargs)[0]

        grad = jax.grad(vec_fun)(param_v)
        hess = jax.hessian(vec_fun)(param_v)

        hess_inv = jnp.linalg.inv(hess)

        # do backtracking to find t
        direction = -hess_inv @ grad
        # reduction amount for the backtracking
        reduction = 0.5*grad.T @ direction
        current_cost = vec_fun(param_v)

        def backtrack_cond(t):
            new_cost = vec_fun(param_v + t*direction)
            exp_cost = current_cost + t*reduction
            return new_cost > exp_cost

        t = jax.lax.while_loop(backtrack_cond,
                               lambda t: self.beta*t, 1)
        new_param_v = param_v + t*direction

        new_param = p_fmt(new_param_v)
        cost, aux = self._cost_fun(new_param, *args, **kwargs)

        if self.terminate:
            solved = self.terminate(cost, aux)
        else:
            eps = jnp.linalg.norm(param_v - new_param_v)
            solved = eps < self.tol

        # old_cost, _ = self._cost_fun(params, *args, **kwargs)
        # jax.debug.print("{} {} old: {} new: {} t: {}, reduction: {}",
        #                 cost, old_cost, params[0], new_param[0], t, reduction)
        return new_param, NewtonState(cost, aux, solved)

# A relaxing solver can be used to implement a barrier-function-based
# solver, among other things
class RelaxingState(NamedTuple):
    # The center solver state
    sub_state: Any
    t: float
    solved: bool

@dataclass
class RelaxingSolver(IterativeSolver):
    sub_solver: IterativeSolver

    init_t: float = 1.
    max_t: float = 100.
    inc_t_factor: float = 1.5

    max_iterations: int = 5000

    def optimality_fun(self, params, state, *args, **kwargs):
        return self.sub_solver.optimality_fun(params, state, *args, t=state.t, **kwargs)

    def init_state(self, params, *args, **kwargs):
        t = float(self.init_t)
        sub_state = self.sub_solver.init_state(params, *args, t=t, **kwargs)
        solved = (sub_state.solved) and (t >= self.max_t)
        return RelaxingState(sub_state, t, solved)

    def update(self, params, state, *args, **kwargs):
        # reinitialize if the sub solver is done
        def adv_t():
            t = jnp.minimum(state.t*self.inc_t_factor, self.max_t)
            sub_state = self.sub_solver.init_state(params, *args, t=t, **kwargs)
            return t, sub_state

        t, sub_state = jax.lax.cond(state.sub_state.solved,
                adv_t, lambda: (state.t, state.sub_state))
        # advance the sub_state
        new_params, sub_state = self.sub_solver.update(params, sub_state, *args, t=t, **kwargs)

        # t_sufficient
        end_of_curve = t >= self.max_t
        solved = jnp.logical_and(sub_state.solved, end_of_curve)
        return new_params, RelaxingState(sub_state, t, solved)