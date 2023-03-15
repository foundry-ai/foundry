import sys
import optax
import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
import jax.scipy as jsp

import stanza.envs
import stanza.util.random
import stanza.policies

from stanza.policies import Rollout, ActionsFeedback
from stanza.util.dataclasses import dataclass, field
from stanza.util import mapped_ravel_pytree

from functools import partial
from typing import Callable

@dataclass(jax=True)
class EstState:
    rng_key: jax.random.PRNGKey
    ref_old_states: jnp.array
    ref_old_gains: jnp.array
    ref_new_states: jnp.array
    ref_new_gains: jnp.array
    total_samples: int

@dataclass(jax=True, kw_only=True)
class FeedbackRollout:
    # A rollout function where extras.A and extras.B
    # are populated
    model_fn: Callable
    rng_key: jax.random.PRNGKey

    samples : int = field(default=0, jax_static=True)
    sigma : float = field(default=0.001)

    use_gains : bool = field(default=True, jax_static=True)
    burn_in : int = field(default=10, jax_static=True)

    # flattened model function
    def _model_fn_flat(self, x_unflat, u_unflat, x_flat, u_flat):
        x = x_unflat(x_flat)
        u = u_unflat(u_flat)
        x = self.model_fn(x, u)
        x_flat, _ = jax.flatten_util.ravel_pytree(x)
        return x_flat

    # returns 
    def _rollout_with_jac(self, flat_model_fn, rng_key, ref_states, ref_gains, state0, actions):
        def rollout(actions):
            return stanza.policies.rollout(
                flat_model_fn, state0, 
                stanza.policies.ActionsFeedback(actions, ref_states, ref_gains)
            ).states
        states = rollout(actions)
        jac = jax.jacrev(rollout)(actions)
        return states, jac

    def _solve_markov(self, W):
        raise NotImplementedError("To be implemented by deriving classes")

    def _solve_A_B(self, jac, prev_gains):
        C = jac[:self.burn_in,self.burn_in:]
        C_k = C[:,:-1]
        C_kp = C[:,1:]
        # flatten out the first dimension into the column dimension
        C_k = jnp.transpose(C_k, (1,2,0,3))
        C_k = C_k.reshape((C_k.shape[0], C_k.shape[1],-1))
        C_kp = jnp.transpose(C_kp, (1,2,0,3))
        C_kp = C_kp.reshape((C_kp.shape[0], C_kp.shape[1],-1))
        # C_k, C_kp are (traj_length - burn_in - 1, x_dim, input x burn_in)
        # the pseudoinverse should broadcast over the first dimension
        # select 1 above the jacobian diagonal for the Bs
        Bs_est = jnp.transpose(jnp.diagonal(jac, 1), (2,0,1))[self.burn_in:]
        # estimate the A matrices
        As_est = C_kp @ jnp.linalg.pinv(C_k) - Bs_est @ prev_gains[self.burn_in:]
    
    def _synth_gains(self, As, Bs):
        return As

    # Modify the actions to account for feedback gains
    def update_actions(self, est_state, state0, actions):
        if not self.use_gains:
            return actions
        state_flat, x_unflat = jax.flatten_util.ravel_pytree(state0)
        actions_flat, actions_unflat, u_unflat = mapped_ravel_pytree(actions)
        model_flat = partial(self._model_fn_flat, x_unflat, u_unflat)

        # We took a step under the system with the old gains,
        # rollout with the old gains
        ref = stanza.policies.rollout(model_flat, state_flat, actions_flat)
        actions_mod = actions_flat + est_state.ref_old_gains @ (ref.states - est_state.ref_states)
        return actions_unflat(actions_mod)

    def __call__(self, est_state, state0, actions):
        state0_flat, x_unflat = jax.flatten_util.ravel_pytree(state0)
        actions_flat, us_unflat, u_unflat = mapped_ravel_pytree(actions)
        model_flat = partial(self._model_fn_flat, x_unflat, u_unflat)

        if est_state is None:
            rng_key = self.rng_key
            ref_states = stanza.policies.rollout_inputs(model_flat, state0_flat, actions_flat).states
            ref_states = jax.lax.stop_gradient(ref_states)
            ref_gains = jnp.zeros((ref_states.shape[0], actions_flat.shape[1], ref_states.shape[1]))
        else:
            rng_key = est_state.rng_key
            ref_states, ref_gains = est_state.ref_new_states, est_state.ref_new_gains

        rng_key, sk = jax.random.split(rng_key)

        states_flat, jac = self._rollout_with_jac(model_flat, sk, ref_states, ref_gains, state0_flat, actions_flat)
        if self.use_gains:
            As, Bs = self._solve_A_B(jac, ref_gains)
            new_gains = self._synth_gains(As, Bs)
            actions_flat = actions_flat + ref_gains @ (states_flat - ref_states)

        actions = us_unflat(actions_flat)
        states = jax.vmap(x_unflat)(states_flat)
        # modify actions based on gains
        new_est_state = EstState(
            rng_key=rng_key, ref_states=ref_states,
            ref_gains=ref_gains
        )
        return new_est_state, Rollout(states, actions)

USE_LEAST_SQUARES = True

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
    
    def inject_gradient(self, est_state, model_fn, states, gains, us, true_jac):
        new_rng, subkey = jax.random.split(est_state.rng)
        W, x_diff = self.rollout(model_fn, subkey, states, gains, us)
        jac = self.calculate_jacobians(W, x_diff, true_jac)
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
        #rollout = partial(ode.envs.rollout_input_gains, self.model_fn, state_0, traj.x, gains)
        rollout = partial(stanza.envs.rollout_input, model_fn, state_0)
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
    
    def calculate_jacobians(self, W, x_diff, true_jac):
        # W: (samples, traj_dim-1, u_dim)
        # x_diff: (samples, traj_dim, x_dim)

        if USE_LEAST_SQUARES:
            T = W.shape[1] + 1
            x_dim = x_diff.shape[-1]
            u_dim = W.shape[2]
            # W: (samples, traj_dim-1, u_dim)
            W = jnp.reshape(W, (W.shape[0], -1))
            # W: (samples, traj_dim-1 * u_dim)
            W = jnp.expand_dims(W, -1)
            # W: (samples, traj_dim-1 * u_dim, 1)
            W_T = jnp.transpose(W, (0, 2, 1))
            # W_T: (samples, 1, traj_dim-1 * u_dim)

            W_W = W @ W_T
            # W_W: (samples, traj_dim-1 * u_dim, traj_dim-1 * u_dim)
            W_W = jnp.sum(W_W, 0)
            # W_W: (traj_dim-1 * u_dim, traj_dim-1 * u_dim)
            x_diff = jnp.expand_dims(x_diff, -1)
            # x_diff (samples, traj_dim, x_dim, 1)

            jac = jnp.zeros((T-1, T, x_dim, u_dim))
            for t in range(1, T):
                M = W_W[:t*u_dim,:t*u_dim]
                # M (t*u_dim, t*u_dim)
                B = x_diff[:, t, ...] @ W_T[:, :, :t*u_dim]
                B = jnp.sum(B, 0).T
                # B (t*u_dim, x_dim)
                X = jsp.linalg.solve(M + 0.00001*jnp.eye(M.shape[0]), B, assume_a='pos')
                # X (t*u_dim, x_dim)
                X = X.reshape((t, u_dim, x_dim))
                X = jnp.transpose(X, (0, 2, 1))
                jac = jac.at[:t,t,...].set(X)

                def print_func(arg, _):
                    X, B, M, t = arg
                    if jnp.any(jnp.isnan(X)) or jnp.any(jnp.isnan(B)) or jnp.any(jnp.isnan(M)):
                        print('t', t)
                        print('X', X)
                        print('B', B)
                        print('M', M)
                        print('s', jnp.linalg.cond(M))
                        import pdb
                        pdb.set_trace()
                        sys.exit(0)
                # jax.experimental.host_callback.id_tap(print_func, (X, B, M,t))
            def print_func(arg, _):
                jac, true_jac, W, x_diff = arg
                if jnp.any(jnp.isnan(jac)):
                    print('---- Computing Jacobian ------')
                    print(jac[1,9,...])
                    print(true_jac[1,9,...])
                    print(jnp.max(jnp.abs(jac - true_jac)))
                    sys.exit(0)
            # jax.experimental.host_callback.id_tap(print_func, (jac, true_jac, W, x_diff))
            return jac
        else:
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
