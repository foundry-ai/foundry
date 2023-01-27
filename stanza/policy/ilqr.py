from functools import partial

import jax
import jax.numpy as jnp
import trajax.optimizers
import stanza.envs

from stanza.logging import logger

class ILQR:
    def __init__(self, x_sample, u_sample,
                cost_fn, model_fn, horizon_length=20, receed=True,
                verbose=False):
        self.u_sample = u_sample
        self.x_sample = x_sample

        self.model_fn = stanza.envs.flatten_model(model_fn, x_sample, u_sample)
        self.cost_fn = stanza.envs.flatten_cost(cost_fn, x_sample, u_sample)
        self.horizon_length = horizon_length
        self.receed = receed
        self.verbose = verbose
    
    def _solve(self, x0, init_us, curr_T):
        x_vec, _ = jax.flatten_util.ravel_pytree(x0)

        def ilqr_cost(x, u, t):
            if self.receed:
                t = t + curr_T
                cost = jax.lax.cond(t > self.horizon_length,
                    lambda: 0.,
                    lambda: 
                        jax.lax.cond(t == self.horizon_length,
                            lambda: self.cost_fn(x, None),
                            lambda: self.cost_fn(x, u))
                )
                return cost
            else:
                return jax.lax.cond(t == self.horizon_length,
                    lambda: self.cost_fn(x, None),
                    lambda: self.cost_fn(x, u))

        _, us, _, grad, _, _, iters = trajax.optimizers.ilqr(
            ilqr_cost,
            lambda x,u,t: self.model_fn(x, u), 
            x_vec, init_us
        )

        # _, us, _, grad, _, _, iters = trajax.optimizers.ilqr(
        #     lambda x,u,t: jax.lax.cond(t == self.horizon_length,
        #         lambda: self.cost_fn(x, None),
        #         lambda: self.cost_fn(x, u)),
        #     lambda x,u,t: self.model_fn(x, u), 
        #     x_vec, init_us
        # )
        if self.verbose:
            logger.info('ilqr grad norm {}, iters {}',
                        jnp.linalg.norm(jax.flatten_util.ravel_pytree(grad)[0]),
                        iters)
        return us

    def init_state(self, x0):
        u_vec, _ = jax.flatten_util.ravel_pytree(self.u_sample)
        us = jnp.zeros((self.horizon_length - 1,) + u_vec.shape)
        us = self._solve(x0, us, 0)
        return us, 0
    
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, state, policy_state):
        us, T = policy_state
        
        _, u_unflatten = jax.flatten_util.ravel_pytree(self.u_sample)
        if self.receed:
            us = us.at[:-1].set(us[1:])
            # re-solve
            us = self._solve(state, us, T)
            return u_unflatten(us[0]), (us, T+1)
        else:
            return u_unflatten(us[T]), (us, T+1)