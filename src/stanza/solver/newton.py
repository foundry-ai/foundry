from stanza.struct import dataclass, replace
from stanza.solver import UnsupportedObectiveError, \
        MinimizeState, Minimize, EqConstraint, IneqConstraint
from stanza.solver.iterative import IterativeSolver
from stanza import Partial

import jax
import jax.experimental.host_callback
import jax.numpy as jnp

DEBUG = False

@dataclass
class NewtonState(MinimizeState):
    # The dual variables for eq constraints
    nu_dual : jnp.array
    # The dual variables for ineq constraints
    lambda_dual : jnp.array

# A newton solver with backtracking support
@dataclass(kw_only=True)
class NewtonSolver(IterativeSolver):
    tol: float = 1e-4
    beta: float = 0.5 # backtracking beta
    alpha: float = 0.005 # backtracking alpha

    eta: float = 0.001 # for interior point

    def init(self, objective, **kwargs):
        if not isinstance(objective, Minimize):
            raise UnsupportedObectiveError("Can only handle unconstrained minimization objectives")
        a, _, b, _, _ = self._constraints(objective,
                    objective.initial_state,
                    objective.initial_params)
        return NewtonState(
            iteration=0,
            solved=False,
            state=objective.initial_state,
            params=objective.initial_params,
            aux=None,
            cost=jnp.zeros(()),
            nu_dual=jnp.zeros_like(a),
            lambda_dual=jnp.ones_like(b)
        )
    
    def _constraints(self, objective, state, params, hess=True):
        eq_resid = []
        eq_grads = []

        ineq_resid = []
        ineq_grads = []
        ineq_hess = []

        v_params, unflatten = jax.flatten_util.ravel_pytree(params)
        for c in objective.constraints:
            if isinstance(c, EqConstraint):
                f = Partial(c.fun, state) if objective.has_state else c.fun
                fun = lambda x: f(unflatten(x))
                eq_resid.append(jnp.atleast_1d(f(v_params)))
                eq_grads.append(jnp.atleast_2d(jax.grad(fun)(v_params)))
            elif isinstance(c, IneqConstraint):
                raise RuntimeError("Currently inequality constraints aren't working for newton solvers!")
                f = Partial(c.fun, state) if objective.has_state else c.fun
                fun = lambda x: f(unflatten(x))
                ineq_resid.append(jnp.atleast_1d(f(v_params)))
                ineq_grads.append(jnp.atleast_2d(jax.grad(fun)(v_params)))
                if hess:
                    ineq_hess.append(jax.hessian(fun)(v_params))

        eq_resid = jnp.concatenate(eq_resid) if eq_resid else jnp.zeros((0,))
        eq_grads = jnp.concatenate(eq_grads) if eq_grads else jnp.zeros((0, v_params.shape[0]))
        ineq_resid = jnp.concatenate(ineq_resid) if ineq_resid else jnp.zeros((0,))
        ineq_grads = jnp.concatenate(ineq_grads) if ineq_grads else jnp.zeros((0, v_params.shape[0]))
        ineq_hess = jnp.sum(jnp.array(ineq_hess), axis=0) if ineq_hess else jnp.zeros((v_params.shape[0],v_params.shape[0]))
        return eq_resid, eq_grads, ineq_resid, ineq_grads, ineq_hess

    def update(self, objective, solver_state):
        if not isinstance(objective, Minimize):
            raise UnsupportedObectiveError("Can only handle unconstrained minimization objectives")
        new_state, _, aux = objective.eval(solver_state.state, solver_state.params)
        # unravel argument structure into param_v
        x, p_fmt = jax.flatten_util.ravel_pytree(solver_state.params)
        nu_dual, lambda_dual = solver_state.nu_dual, solver_state.lambda_dual

        vec_cost = lambda v: objective.eval(solver_state.state, p_fmt(v))[1]
        f_cost = vec_cost(x)
        f_grad = jax.grad(vec_cost)(x)
        f_hess = jax.hessian(vec_cost)(x)
            
        r_primal, A, f, D, ineq_hess = self._constraints(objective,
                                                solver_state.state, solver_state.params)
        r_cent = -lambda_dual*f - self.eta
        r_dual = f_grad + D.T @ lambda_dual + A.T @ nu_dual


        # The off-diagonal zeros matrices
        z1 = jnp.zeros((lambda_dual.shape[0], A.shape[0]))
        z2 = jnp.zeros((A.shape[0], A.shape[0]))
        # The big boy matrix
        ld = (lambda_dual * D) if lambda_dual.shape[0] > 0 else jnp.zeros((0,D.shape[1]))
        M = jnp.block([
            [f_hess + ineq_hess,    D.T,                    A.T],
            [ld,       -jnp.diag(f), z1],
            [A,                     z1.T,                   z2]
        ])
        #jax.debug.print("M: {}", M)
        r = jnp.block([r_dual, r_cent, r_primal])
        # r_norm_sq = jnp.sum(jnp.square(r))
        #jax.debug.print("r: {}", r)
        d = jnp.linalg.solve(M, -r)
        #jax.debug.print("d: {}", d)
        # split d into dx, dlambda, dnu
        dx, dlambda, dnu = d[:x.shape[0]], \
                d[x.shape[0]:x.shape[0] + lambda_dual.shape[0]], \
                d[x.shape[0] + lambda_dual.shape[0]:]

        # do backtracking
        #jax.debug.print("l: {}", -lambda_dual/dlambda)
        ms = jnp.where(dlambda < 0, -lambda_dual/dlambda, 1)
        s_max = jnp.min(0.99*ms, initial=1)

        if False:
            # detect if we are in a locally-concave area
            # and step the opposite direction
            dx = -jnp.sign(jnp.dot(f_grad, dx))*dx
            dec = jnp.dot(f_grad, dx)
            def backtrack_cond(s):
                #r_primal, _, f, _, _ = self._constraints(objective, solver_state.state, p_fmt(x + dx), False)
                # L_new = vec_cost(x + s*dx) + jnp.dot(nu_dual + s*dnu, r_primal) \
                #         + jnp.dot(lambda_dual + s*dlambda,f)
                cost_new = vec_cost(x + s*dx)
                jax.debug.print("tried {}: {}", s, cost_new)
                return cost_new > f_cost + self.alpha*s*dec/2
                #return L_new > L + self.alpha*s*jnp.dot(r_dual, dx)/2
            s = jax.lax.while_loop(backtrack_cond,
                                lambda s: self.beta*s, s_max)
        elif True:
            #dx = -jnp.sign(jnp.dot(f_grad, dx))*dx
            #dec = jnp.dot(f_grad, dx)
            dec = jnp.dot(r_dual, dx)
            if DEBUG:
                jax.debug.print("dec {}", dec)
            L = vec_cost(x) + jnp.dot(nu_dual, r_primal) \
                    + jnp.dot(lambda_dual,f)
            def backtrack_cond(s):
                r_primal, _, f, _, _ = self._constraints(objective, solver_state.state, p_fmt(x + dx), False)
                L_new = vec_cost(x + s*dx) + jnp.dot(nu_dual + s*dnu, r_primal) \
                        + jnp.dot(lambda_dual + s*dlambda,f)
                tol = self.alpha*s*dec/2
                if DEBUG:
                    jax.debug.print("tried {}: {} ({} tol {})", s, L_new, L, tol)
                return L_new > L + tol
            s = jax.lax.while_loop(backtrack_cond,
                                lambda s: self.beta*s, s_max)

        new_x = x + s*dx
        new_nu_dual = nu_dual + s*dnu
        new_lambda_dual = lambda_dual + s*dlambda

        if DEBUG:
            jax.debug.print("x: {} nu: {} lambda: {}", x, nu_dual, lambda_dual)
            jax.debug.print("cost: {}", f_cost)
            jax.debug.print("dx: {}", dx)
            jax.debug.print("dec: {}", dec)
            jax.debug.print("dlambda: {}", dlambda)
            jax.debug.print("dnu: {}", dnu)
            jax.debug.print("step: {}", s)
            jax.debug.print("up: {}", jnp.linalg.norm(s*dx))
        # def f(arg, _):
        #     if jnp.any(jnp.isnan(arg)):
        #         print(arg)
        #         import sys
        #         sys.exit(0)
        # jax.experimental.host_callback.id_tap(f, new_x)

        # Find the new state
        new_params = p_fmt(new_x)
        solved = jnp.linalg.norm(new_x - x) < self.tol
        solved = jnp.logical_or(solved, jnp.any(jnp.isnan(new_x)))
        return NewtonState(
            iteration=solver_state.iteration + 1,
            solved=solved,
            params=new_params,
            state=new_state,
            aux=aux,
            cost=f_cost,
            nu_dual=new_nu_dual,
            lambda_dual=new_lambda_dual
        )