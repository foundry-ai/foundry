import importlib
import inspect

import jax

from jinx.random import PRNGDataset
from jinx.dataset import MappedDataset

# Generic environment
class Environment:
    @property
    def action_size(self):
        return None

    def reset(self, key):
        pass

    def step(self, state, action):
        pass

    def observe(self, state, name):
        pass
    
# Helper function to do rollouts with
def rollout(env, length, policy, key):
    def scan_fn(comb_state, _):
        env_state, policy_state, u = comb_state
        new_env_state = env.step(env_state, u)

        new_u, new_policy_state = policy(new_env_state, policy_state)
        return (new_env_state, new_policy_state, new_u), (env_state, u)

    # Do the first step manually to populate the policy state
    env_state = env.reset(key)
    u, policy_state = policy(env_state)
    init_state = env_state, policy_state, u

    _, xu = jax.lax.scan(scan_fn, init_state, None, length=length)
    return xu

# Global registry

__ENV_BUILDERS = {}

def create(name, *args, **kwargs):
    # register buildres if empty
    builder = __ENV_BUILDERS[name]()
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

class EnvironmentDataset(MappedDataset):
    def __init__(self, key, env, policy, observations, trajectory_length):
        self._env = env
        self._policy = policy
        self._observations = observations

        dataset = PRNGDataset(key)
        super(dataset, self._gen_trajectory)

    def _gen_trajectory(self, key):
        def do_step():
            self._policy(self._env, state)
            pass
        init_state = self.env.reset(key)
        init_policy_state = self._policy(self._env, state, None)