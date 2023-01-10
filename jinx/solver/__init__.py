import jax
import jax.flatten_util
import jax.numpy as jnp

import optax

from typing import NamedTuple, Any, Callable
from dataclasses import dataclass

class SolverResults(NamedTuple):
    solved: bool
    final_params: Any
    final_state: Any
    history: Any

import jinx.solver.implicit_diff as idf

class IterativeSolver:
    # Must have 'max_iterations' attribute
    #
    # The solver state must have a "solved" boolean


    # To be overridden by the iterative solver
    def init_state(self, params, *args, **kwargs):
        raise NotImplementedError("Must be implemented")

    def update(self, params, state, *args, **kwargs):
        raise NotImplementedError("Must be implemented")

    def _do_step(self, params, opt_state, args, kwargs):
        new_params, new_state = self.update(params, opt_state, *args, **kwargs)
        return new_params, new_state

    def _scan_fun(self, state, _):
        (params, opt_state), (args, kwargs) = state
        # Only do a step if we are not done yet
        new_params, new_state = jax.lax.cond(opt_state.solved,
            lambda a, b, _0, _1: (a, b),
            self._do_step, 
            params, opt_state, args, kwargs)

        state = (new_params, new_state), (args, kwargs)
        return state, opt_state


    # Suitable for wrapping with implicit diffs
    def _run(self, init_params, args, kwargs):
        init_opt_state = self.init_state(init_params, *args, **kwargs)
        scan_state = (init_params, init_opt_state), (args, kwargs)

        scan_state, history = \
            jax.lax.scan(self._scan_fun, scan_state,
                        None, length=self.max_iterations)
        (final_params, final_state), _ = scan_state
        return final_params, final_state, history

    def _run_optimality_fun(self, params_state, args, kwargs):
        params, state = params_state
        return self.optimality_fun(params, state, *args, **kwargs)

    def run(self, init_params, *args, **kwargs):
        run = self._run
        if getattr(self, "implicit_diff", True) and \
                getattr(self, 'optimality_fun', None) is not None:
            decorator = idf.custom_root(self._run_optimality_fun, has_aux=True,
                                        solve=None)
            run = decorator(run)
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
    max_iterations: int = 500

    def _cost_fun(self, params, *args, **kwargs):
        cost, aux = (self.fun(params, *args, **kwargs)
            if self.has_aux else
            self.fun(params, *args, **kwargs), None
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
            solved = self.terminate(params, new_params, cost)
        else:
            solved = False

        return new_params, OptaxState(cost, aux, opt_state, solved)

class NewtonState(NamedTuple):
    cost: float
    aux: Any
    solved: bool

@dataclass
class NewtonSolver(IterativeSolver):
    fun: Callable
    has_aux: bool = False
    tol: float = 1e-3
    max_iterations: int = 500
    # for damped newton steps
    damping: float = 0

    # Will handle both the has_aux and not-has_aux cases
    def _cost_fun(self, params, *args, **kwargs):
        cost, aux = (self.fun(params, *args, **kwargs)
            if self.has_aux else
            self.fun(params, *args, **kwargs), None
        )
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

        local_norm = grad.T @ hess_inv @ grad
        f = 1/(1 + self.damping*jnp.sqrt(local_norm))

        new_param_v = param_v - f*jnp.linalg.inv(hess) @ grad

        eps = jnp.linalg.norm(new_param_v - param_v, ord=2)
        solved = eps < self.tol

        new_param = p_fmt(new_param_v)
        cost, aux = self._cost_fun(new_param, *args, **kwargs)
        old_cost, _ = self._cost_fun(params, *args, **kwargs)
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

    init_t: float = 0.01
    max_t: float = 100
    inc_t_factor: float = 2

    max_iterations: int = 500

    def optimality_fun(self, params, state, *args, **kwargs):
        return self.sub_solver.optimality_fun(params, state, *args, t=state.t, **kwargs)

    def init_state(self, params, *args, **kwargs):
        t = self.init_t
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
        end_of_curve = state.t >= self.max_t
        solved = jnp.logical_and(sub_state.solved, end_of_curve)
        return new_params, RelaxingState(sub_state, t, solved)