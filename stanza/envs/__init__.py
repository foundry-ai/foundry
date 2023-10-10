# Generic environment. Note that all
# environments are also adapters
class Environment:
    def sample_state(self, rng_key):
        raise NotImplementedError("Must impelement sample_state()")

    def sample_action(self, rng_key):
        raise NotImplementedError("Must impelement sample_action()")

    def reset(self, key):
        raise NotImplementedError("Must impelement reset()")

    # rng_key may be None. if it is None and the environment
    # is stochastic, throw an error!
    def step(self, state, action, rng_key):
        raise NotImplementedError("Must impelement step()")

    def observe(self, state):
        return state
    
    def reward(self, state, action, next_state):
        raise NotImplementedError("Must impelement reward()")
    
    def render(self, state, *, 
                    # every environment *must* support "image"
                    # without any additional kwargs
                    width=256, height=256, mode="image",
                    # the kwargs can be extra things we may want
                    # to render. Environments should just ignore kwargs
                    # they do not support
                    **kwargs):
        raise NotImplementedError("Must implement render()")

from stanza.envs.builders import create, register_lazy

register_lazy('pusht', '.pusht')
register_lazy('pendulum', '.pendulum')
register_lazy('linear', '.linear')
register_lazy('quadrotor', '.quadrotor')
register_lazy('gym', '.gymnasium')
register_lazy('gymnax', '.gymnax')
register_lazy('brax', '.brax')
register_lazy('robosuite', '.robosuite')