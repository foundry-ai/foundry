from stanza.util.dataclasses import dataclass, replace
from stanza.solver import IterativeSolver, UnsupportedObectiveError, \
        MinimizeState, Minimize, EqConstraint, IneqConstraint
from stanza import Partial

import jax
import jax.numpy as jnp

@dataclass(jax=True)
class NewtonState(MinimizeState):
    # The dual variables for eq constraints
    nu_dual : jnp.array
    # The dual variables for ineq constraints
    lambda_dual : jnp.array

# A newton solver with backtracking support
@dataclass(jax=True, kw_only=True)
class NewtonSolver(IterativeSolver):
    tol: float = 1e-2
    beta: float = 0.5 # backtracking beta

    def init_state(self, objective):
        a, _ = self._eq_constraints(objective,
                    objective.initial_state,
                    objective.initial_params)
        b, _, _ = self._ineq_constraints(objective,
                    objective.initial_state,
                    objective.initial_params)
        return NewtonState(
            iteration=0,
            solved=False,
            state=objective.initial_state,
            params=objective.initial_params,
            aux=None,
            nu_dual=jnp.zeros_like(a),
            lambda_dual=jnp.zeros_like(b)
        )
    
    # gradient of the objective at params == 0
    def optimality(self, objective, solver_state):
        grad = jax.grad(lambda p: objective.eval(solver_state.state, p)[1])(solver_state.params)
        return grad
    
    def _eq_constraints(self, objective, state, params):
        grads = []
        vals = []
        for c in objective.constraints:
            if isinstance(c, EqConstraint):
                f = lambda x: Partial(c.fun, state) if objective.has_state else c.fun
                v_params, unflatten = jax.flatten_util.ravel_pytree(params)
                fun = lambda x: f(unflatten(x))
                grads.append(f(params))
                vals.append(jax.grad(fun)(v_params))
            else:
                raise UnsupportedObectiveError("Cannot handle inequality constraints (yet)")
        return jnp.concatenate(vals), jnp.concatenate(grads)

    def _ineq_constraints(self, objective, state, params):
        hess = []
        grads = []
        vals = []
        for c in objective.constraints:
            if isinstance(c, EqConstraint):
                f = lambda x: Partial(c.fun, state) if objective.has_state else c.fun
                v_params, unflatten = jax.flatten_util.ravel_pytree(params)
                fun = lambda x: f(unflatten(x))
                grads.append(f(params))
                vals.append(jax.grad(fun)(v_params))
                hess.append(jax.grad(jax.grad(fun))(v_params))
        return jnp.concatenate(vals), jnp.concatenate(grads), \
                jnp.sum(jnp.array(hess), axis=0)

    def update(self, objective, solver_state):
        if not isinstance(objective, Minimize):
            raise UnsupportedObectiveError("Can only handle unconstrained minimization objectives")
        if solver_state is None:
            solver_state = self.init_state(objective)
        new_state, _, aux = objective.eval(solver_state.state, solver_state.params)

        # unravel argument structure into param_v
        x, p_fmt = jax.flatten_util.ravel_pytree(solver_state.params)
        nu_dual, lambda_dual = solver_state.nu_dual, solver_state.lambda_dual

        vec_cost = lambda v: objective.eval(solver_state.state, p_fmt(v))[1]
        f_grad = jax.grad(vec_cost)(x)
        f_hess = jax.hessian(vec_cost)(x)
        r_primal, A = self._eq_constraints(objective, solver_state.state, solver_state.params)
        r_cent, D, ineq_hess = self._ineq_constraints(objective, solver_state.state, solver_state.params)


        # The 
        M = jnp.block([
            [f_hess + ineq_hess, D.T, A.T],
            [lambda_dual * D, -jnp.diag(lambda_dual), jnp.zeros(lambda_dual.shape[0], A.shape[0])]
            [A, jnp.zeros((A.shape[0], A.shape[0]))]
        ])
        r_dual = f_grad + A.T @ solver_sta
        v = jnp.block([f_grad, r_primal])

        direction = jnp.solve()

        reduction = 0.5*grad.T @ direction
        current_cost = vec_cost(param_v)
        def backtrack_cond(t):
            new_cost = vec_cost(param_v + t*direction)
            exp_cost = current_cost + t*reduction
            return new_cost > exp_cost

        t = jax.lax.while_loop(backtrack_cond,
                            lambda t: self.beta*t, 1)
        new_param_v = param_v + t*direction
        eps = jnp.linalg.norm(param_v - new_param_v)

        # Find the new state
        new_params = p_fmt(new_param_v)
        solved = eps < self.tol
        return MinimizeState(
            iteration=solver_state.iteration + 1,
            solved=solved,
            params=new_params,
            state=new_state,
            aux=aux
        )