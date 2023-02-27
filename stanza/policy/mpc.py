import jax
import jax.numpy as jnp

import stanza.policy

from typing import Any

from stanza import Partial

from stanza.util.logging import logger
from stanza.util.dataclasses import dataclass, replace, field

from stanza.solver import Solver
from stanza.solver.newton import NewtonSolver
from stanza.policy import Actions, PolicyOutput

def _centered_log(sdf, *args):
    u_x, fmt = jax.flatten_util.ravel_pytree(args)
    # calculate gradient at zero
    grad = jax.jacrev(lambda v_x: -jnp.log(-sdf(*fmt(v_x))))(jnp.zeros_like(u_x))
    grad_term = grad @ u_x
    barrier = -jnp.log(-sdf(*args)) - grad_term
    return barrier - grad_term

# A vanilla MPC controller
@dataclass(jax=True)
class MPC:
    action_sample: Any
    cost_fn : Any
    model_fn : Any

    # Horizon is part of the static jax type
    horizon_length : int = field(default=20, jax_static=True)
    # Solver must be a dataclass with a "fun" argument
    solver : Solver = NewtonSolver()

    # TODO: Make these actually do things
    # If the horizon should receed or stay static
    receed : bool = field(default=True, jax_static=True)
    # If the controller should replan at every iteration
    replan : bool = field(default=True, jax_static=True)

    def _loss_fn(self, state0, actions):
        r = stanza.policy.rollout(
                self.model_fn, state0, policy=Actions(actions)
            )
        return self.cost_fn(r.states, r.actions)

    def __call__(self, state, policy_state=None):
        if policy_state is None:
            actions_flat, _ = jax.flatten_util.ravel_pytree(self.action_sample)
            actions_flat = jnp.zeros((self.horizon_length - 1, actions_flat.shape[0]))
        else:
            actions_flat = policy_state

        # shift the us by 1
        actions_flat = actions_flat.at[:-1].set(actions_flat[1:])
        _, unflatten = jax.flatten_util.ravel_pytree(self.action_sample)
        actions = jax.vmap(unflatten)(actions_flat)

        # update the solver target function
        solver = replace(self.solver, fun=Partial(self._loss_fn, state))
        res = solver.run(init_params=actions)
        actions = res.params
        action0 = jax.tree_util.tree_map(lambda x: x[0], actions)
        return PolicyOutput(action0)