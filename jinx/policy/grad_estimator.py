import sys
import optax
import jax
import jax.numpy as jnp
import jax.tree_util as tree_util

import jinx.envs
import jinx.util

from functools import partial
from typing import NamedTuple, Any

class EstimatorState(NamedTuple):
    rng: jax.random.PRNGKey
    total_samples: int

class IsingEstimator:
    def __init__(self, rng_key, samples, sigma):
        self.rng_key = rng_key
        self.samples = samples
        self.sigma = sigma

        @jax.custom_vjp
        def _inject_gradient(xs, us, jac):
            return xs

        def _inject_gradient_fwd(xs, us, jac):
            return xs, jac

        def _inject_gradient_bkw(res, g):
            jac = res
            return (None, self.bkw(jac, g), None)

        _inject_gradient.defvjp(_inject_gradient_fwd, _inject_gradient_bkw)
        
        self._inject_gradient = _inject_gradient
    
    def init(self):
        return EstimatorState(
            rng=self.rng_key,
            total_samples=0
        )
    
    def inject_gradient(self, est_state, model_fn, states, gains, us):
        new_rng, subkey = jax.random.split(est_state.rng)
        W, x_diff = self.rollout(model_fn, subkey, states, gains, us)
        jac = self.calculate_jacobians(W, x_diff)
        x = self._inject_gradient(states, us, jac)
        return EstimatorState(
            rng=new_rng,
            total_samples=est_state.total_samples + W.shape[0]
        ), jac, x
    
    # the forwards step
    def rollout(self, model_fn, rng, traj, gains, us):
        rng = self.rng_key if rng is None else rng
        state_0 = jax.tree_util.tree_map(lambda x: x[0], traj)

        # do a bunch of rollouts
        W = self.sigma*jax.random.choice(rng, jnp.array([-1,1]), (self.samples,) + us.shape)
        # rollout all of the perturbed trajectories
        #rollout = partial(jinx.envs.rollout_input_gains, self.model_fn, state_0, traj.x, gains)
        rollout = partial(jinx.envs.rollout_input, model_fn, state_0)
        # Get the first state
        trajs = jax.vmap(rollout)(us + W)
        # subtract off x_bar
        x_diff = trajs - traj
        def print_func(arg, _):
            W, us, gains, x_diff, traj, trajs = arg
            print('---- Rolling out Trajectories ------')
            print('W_nan', jnp.any(jnp.isnan(W)))
            print('x_diff_nan', jnp.any(jnp.isnan(x_diff)))
            print('us_nan', jnp.any(jnp.isnan(us)))
            print('gains_nan', jnp.any(jnp.isnan(gains)))
            print('traj_nan', jnp.any(jnp.isnan(traj)))
            print('trajs_nan', jnp.any(jnp.isnan(trajs)))
            print('us_max', jnp.max(us))
            if jnp.any(jnp.isnan(x_diff)):
                sys.exit(0)
        # jax.experimental.host_callback.id_tap(print_func, (W, us, gains, x_diff, traj, trajs))
        return W, x_diff
    
    def calculate_jacobians(self, W, x_diff):
        # W: (samples, traj_dim-1, u_dim)
        # x_diff: (samples, traj_dim, x_dim)
        W = jnp.expand_dims(W, -2)
        W = jnp.tile(W, [1, 1, x_diff.shape[1], 1])
        # W: (samples, traj_dim-1, traj_dim, u_dim)
        x_diff = jnp.expand_dims(x_diff, -3)
        # x_diff: (samples, 1,  traj_dim, x_dim)

        W = jnp.expand_dims(W, -2)
        x_diff = jnp.expand_dims(x_diff, -1)
        # W: (samples, traj_dim - 1, traj_dim, 1, u_dim)
        # x_diff: (samples, 1, traj_dim, x_dim, 1)
        jac = jnp.mean(x_diff @ W, axis=0)/(self.sigma*self.sigma)
        # jac: (traj_dim-1, traj_dim, x_dim, u_dim)
        # (u,v) entry contains the jacobian from time u to state v

        # we need to zero out at and below the diagonal
        # (there should be no correlation, but just in case)
        tri = jax.numpy.tri(jac.shape[0], jac.shape[1], dtype=bool)
        tri = jnp.expand_dims(jnp.expand_dims(tri, -1),-1)
        tri = jnp.tile(tri, [1,1,jac.shape[2], jac.shape[3]])

        # fill lower-triangle with zeros
        jac = jnp.where(tri, jnp.zeros_like(jac), jac)
        def print_func(arg, _):
            jac, W, x_diff = arg
            print('---- Computing Jacobian ------')
            print(x_diff.shape)
            print('x_diff', x_diff[0])
            print('W_nan', jnp.any(jnp.isnan(W)))
            print('x_diff_nan', jnp.any(jnp.isnan(x_diff)))
            print('jac_nan', jnp.any(jnp.isnan(jac)))
            if jnp.any(jnp.isnan(jac)):
                sys.exit(0)
        # jax.experimental.host_callback.id_tap(print_func, (jac, W, x_diff))
        return jac
    
    # the backwards step
    def bkw(self, jac, g):
        jac_T = jnp.transpose(jac, (0,1,3,2))
        # (traj_dim, traj_dim, u_dim, x_dim) @ (1, traj_dim, x_dim, 1)
        grad = jac_T @ jnp.expand_dims(jnp.expand_dims(g, -1),0)
        # grad: (traj_dim, traj_dim, u_dim, 1)
        # sum over columns to combine all transitions for a given time
        grad = jnp.sum(jnp.squeeze(grad,-1), 1)
        return grad
