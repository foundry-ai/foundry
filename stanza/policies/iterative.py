import jax.numpy as jnp
import optax
import jax
import sys

import stanza.policies as policies
from stanza.policies import PolicyOutput, Trajectory

from stanza.util.dataclasses import dataclass, field, replace
from typing import NamedTuple, Callable, Any
from functools import partial
from stanza import Partial

DEBUG = False

@dataclass(jax=True)
class FeedbackState:
    gains: Any
    nominal_states: Any
    # The last A, B estimates
    # from the last rollout
    As_est: Any
    Bs_est: Any
    est_state: Any

@dataclass(jax=True)
class EstimatorRollout:
    model_fn : Callable = None
    grad_estimator : Callable = None

    def _rollout(self, state0, actions):
        rollout = policies.rollout_inputs(self.model_fn, state0, actions)
        return rollout.states
    
    @property
    def has_state(self):
        return True

    def __call__(self, roll_state, state0, actions):
        action0 = jax.tree_util.tree_map(lambda x: x[0], actions)
        rollout_ = Partial(self._rollout, state0)
        if self.grad_estimator is not None:
            rollout = use_estimator(self.grad_estimator, rollout_)
        else: 
            rollout = lambda s, a: (s, rollout_(a))
        roll_state, states = rollout(roll_state, actions)
        return roll_state, Trajectory(
            states=states,
            actions=actions
        )

# An MPC with feedback gains, but no barrier functions
@dataclass(jax=True)
class FeedbackRollout:
    model_fn : Callable = None
    burn_in : int = field(jax_static=True, default=10)
    Q_coef : float = 10
    R_coef : float = 1
    grad_estimator : Callable = None
    
    @property
    def has_state(self):
        return True

    def _flat_model(self, state_sample, action_sample, state, action, rng_key):
        _, state_uf = jax.flatten_util.ravel_pytree(state_sample)
        _, action_uf = jax.flatten_util.ravel_pytree(action_sample)
        state = state_uf(state)
        action = action_uf(action)
        state = self.model_fn(state, action, rng_key)
        state, _ = jax.flatten_util.ravel_pytree(state)
        return state

    def _solve_dynamics(self, jac, prev_gains):
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
        return As_est, Bs_est

    def _compute_gains(self, As_est, Bs_est, prev_gains):
        # synthesize new gains
        # Q = 0.001*jnp.eye(jac.shape[-2])
        # R = 100*jnp.eye(jac.shape[-1])
        Q = self.Q_coef*jnp.eye(As_est.shape[-2])
        R = self.R_coef*jnp.eye(Bs_est.shape[-1])
        def gains_recurse(P_next, AB):
            A, B = AB
            M = R + B.T @ P_next @ B

            N = A.T @ P_next @ B
            F = jnp.linalg.inv(M) @ N.T
            P = A.T @ P_next @ A - N @ F + Q

            P_scale = 1/(1 + jnp.linalg.norm(P, ord='fro')/100)
            P = P * P_scale
            return P, (F, P)

        _, (gains_est, Ps) = jax.lax.scan(gains_recurse, Q, (As_est, Bs_est), reverse=True)
        gains_est = -gains_est
        new_gains = prev_gains.at[self.burn_in:].set(gains_est)

        if DEBUG:
            def print_fun(args, _):
                prev_gains, new_gains, As_cl, As_est, Bs_est, C_k, Ps, jac = args
                # if jnp.any(jnp.isnan(new_gains)) \
                #         or not jnp.all(jnp.isfinite(new_gains)):
                if True:
                    s = jnp.linalg.svd(C_k, compute_uv=False)
                    sv = lambda s: jnp.max(jnp.linalg.svd(s, compute_uv=False))
                    eig = lambda s: jnp.max(jnp.linalg.eig(s)[0])

                    As_cl_sv = jnp.linalg.svd(As_cl, compute_uv=False)
                    As_sv = jnp.linalg.svd(As_est, compute_uv=False)

                    max_eig = jnp.mean(jax.vmap(eig)(As_cl))
                    # if max_eig > 1:
                    if True:
                        print('gain_sv', jax.vmap(sv)(new_gains))
                        print('As_cl_eig', jax.vmap(eig)(As_cl))
                        print('As_eig', jax.vmap(eig)(As_est))
                        print('As_cl_sv', jax.vmap(sv)(As_cl))
                        print('As_sv', jax.vmap(sv)(As_est))
                        print('Ps_sv', jax.vmap(sv)(Ps))
                        print('s_diff_full', As_cl_sv - As_sv)
                        # sys.exit(0)
            As_cl = As_est + Bs_est @ gains_est
            jax.experimental.host_callback.id_tap(print_fun, (prev_gains, new_gains, \
                As_cl, As_est, Bs_est, C_k, Ps, jac))
        return new_gains

    # rollout new nominal trajectory with the stepped actions
    # compute new gains, and modify roll_state to include new gains + actions
    def update_actions(self, roll_state, state0, actions):
        action0 = jax.tree_util.tree_map(lambda x: x[0], actions)
        flat_model = Partial(self._flat_model, state0, action0)

        state0_flat, state_uf = jax.flatten_util.ravel_pytree(state0)
        action0_flat, action_uf = jax.flatten_util.ravel_pytree(action0)
        actions_flat = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])(actions)

        new_states = policies.rollout(flat_model, state0_flat, 
            policy=FeedbackActions(roll_state.gains, 
                roll_state.nominal_states, actions_flat)).states
        # adjust the actions under the old gains to be centered
        # around the new nominal trajectories
        states_earlier = jax.tree_util.tree_map(lambda x: x[:-1], new_states)
        actions = jax.vmap(gain_action)(actions_flat, roll_state.gains, 
                    roll_state.nominal_states, states_earlier)
        # compute new gains
        new_gains = self._compute_gains(roll_state.As_est, roll_state.Bs_est, roll_state.gains)
        roll_state = replace(roll_state,
            nominal_states=states_earlier,
            gains=new_gains
        )
        actions = jax.vmap(action_uf)(actions)
        return roll_state, actions
    
    def __call__(self, roll_state, state0, actions):
        action0 = jax.tree_util.tree_map(lambda x: x[0], actions)
        state0_flat, state_uf = jax.flatten_util.ravel_pytree(state0)
        action0_flat, action_uf = jax.flatten_util.ravel_pytree(action0)
        actions_flat = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])(actions)
        flat_model = Partial(self._flat_model, state0, action0)

        if roll_state is None:
            length, state_dim, action_dim = actions_flat.shape[0], \
                                state0_flat.shape[0], \
                                action0_flat.shape[0]
            roll_state = FeedbackState(
                gains=jnp.zeros((length, action_dim, state_dim)),
                nominal_states=jnp.zeros((length, state_dim)),
                As_est=jnp.zeros((length, state_dim, state_dim)),
                Bs_est=jnp.zeros((length, state_dim, action_dim)),
                est_state=None,
            )
        rollout = lambda actions: policies.rollout(flat_model, state0_flat,
                            policy=FeedbackActions(
                                roll_state.gains,
                                roll_state.nominal_states,
                                actions
                            )
                        ).states

        if self.grad_estimator is not None:
            est_state, jac, states = self.grad_estimator(rollout)(
                roll_state.est_state, actions_flat
            )
            As_est, Bs_est = self._solve_dynamics(jac, roll_state.gains)
        else:
            est_state = None
            states = rollout(actions_flat)
            states_earlier = jax.tree_util.tree_map(lambda x: x[:-1], states)
            jac_fn = jax.jacrev(flat_model, argnums=(0,1))
            As_est, Bs_est = jax.vmap(jac_fn)(states_earlier, actions_flat, None)
            As_est, Bs_est = As_est[self.burn_in:], Bs_est[self.burn_in:]

        # modify the actions to also include the current gains
        states_earlier = jax.tree_util.tree_map(lambda x: x[:-1], states)
        actions_flat = jax.vmap(gain_action)(actions_flat, roll_state.gains, 
                        roll_state.nominal_states, states_earlier)
        roll_state = replace(roll_state,
            est_state=est_state,
            As_est=As_est, Bs_est=Bs_est
        )
        actions = jax.vmap(action_uf)(actions_flat)
        states = jax.vmap(state_uf)(states)
        return roll_state, Trajectory(states=states, actions=actions)
        
def gain_action(action, gain, nominal_state, state):
    action_flat, action_uf = jax.flatten_util.ravel_pytree(action)
    state_flat, _ = jax.flatten_util.ravel_pytree(state)
    nominal_state_flat, _ = jax.flatten_util.ravel_pytree(nominal_state)
    action_flat = action_flat + gain @ (state_flat - nominal_state_flat)
    action = action_uf(action_flat)
    return action

@dataclass(jax=True)
class FeedbackActions:
    gains: Any
    nominal_states: Any
    actions: Any

    @property
    def rollout_length(self):
        lengths, _ = jax.tree_util.tree_flatten(
            jax.tree_util.tree_map(lambda x: x.shape[0], self.actions)
        )
        return lengths[0] + 1

    @jax.jit
    def __call__(self, input):
        T = input.policy_state if input.policy_state is not None else 0
        action = jax.tree_util.tree_map(lambda x: x[T], self.actions)
        gain = jax.tree_util.tree_map(lambda x: x[T], self.gains)
        nominal_state = jax.tree_util.tree_map(lambda x: x[T], self.nominal_states)
        action = gain_action(action, gain, nominal_state, input.observation)
        return PolicyOutput(
            action=action,
            policy_state=T + 1
        )