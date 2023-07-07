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