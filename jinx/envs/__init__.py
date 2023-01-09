import importlib
import inspect

import jax

from jinx.random import PRNGDataset
from jinx.dataset import MappedDataset
from jinx.util import scan_unrolled, tree_append

# Generic environment
class Environment:
    def sample_action(self, rng_key):
        raise NotImplementedError("Must impelement sample_action()")

    def reset(self, key):
        raise NotImplementedError("Must impelement reset()")

    def step(self, state, action):
        raise NotImplementedError("Must impelement step()")
    
    # The following two are optional
    def cost(self, states, actions):
        raise NotImplementedError("Must impelement cost()")
    
# Helper function to do rollouts with
def rollout_policy(model_fn, state0, length, policy,
                    policy_state=None,
                    ret_policy_state=False):
    if hasattr(policy, 'init_state'):
        policy_state = policy.init_state

    def scan_fn(comb_state, _):
        env_state, policy_state = comb_state
        if policy_state is not None:
            u, new_policy_state = policy(env_state, policy_state)
        else:
            u = policy(env_state)
            new_policy_state = None
        new_env_state = model_fn(env_state, u)
        return (new_env_state, new_policy_state), (env_state, u)

    # Do the first step manually to populate the policy state
    state = (state0, policy_state)
    (state_f, ef), (states, us) = jax.lax.scan(scan_fn, state, None, length=length-1)

    states = tree_append(states, state_f)
    if ret_policy_state:
        return states, us, ef
    else:
        return states, us

# Global registry
def rollout_input(model_fn, state_0, us):
    def scan_fn(state, u):
        new_state = model_fn(state, u)
        return new_state, state
    final_state, states = jax.lax.scan(scan_fn, state_0, us)
    states = tree_append(states, final_state)
    return states


def rollout_input_gains(model_fn, state_0, ref_xs, ref_gains, us):
    def scan_fn(state, i):
        ref_x, ref_gain, u = i
        new_state = model_fn(state, u + ref_gain @ (state.x - ref_x))
        return new_state, state
    final_state, states = jax.lax.scan(scan_fn, state_0, (ref_xs[:-1], ref_gains, us))
    states = tree_append(states, final_state)
    return states

__ENV_BUILDERS = {}

def create(type, *args, **kwargs):
    # register buildres if empty
    builder = __ENV_BUILDERS[type]()
    return builder(*args, **kwargs)

# Register them lazily so we don't
# import dependencies we don't actually use
# i.e the appropriate submodule will be imported
# for the first time during create()
def register_lazy(name, module_name):
    frm = inspect.stack()[1]
    pkg = inspect.getmodule(frm[0]).__name__
    def make_env_constructor():
        mod = importlib.import_module(module_name, package=pkg)
        builder = mod.builder()
        return builder
    __ENV_BUILDERS[name] = make_env_constructor

register_lazy('brax', '.brax')
register_lazy('gym', '.gym')
register_lazy('pendulum', '.pendulum')
register_lazy('linear', '.linear')