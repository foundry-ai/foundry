class PendulumEnvironment(Environment):
    def __init__(self):
        pass
    
    @property
    def action_size(self):
        return 1

    def reset(self, key):
        # pick random position between +/- radians from right
        pos = jax.random.uniform(key,shape=(1,), minval=-1,maxval=1)
        vel = jax.random.zeros((1,))
        state = jax.random.stack((pos, vel))
        return state

    def step(self, state, action):
        return state

def builder():
    return PendulumEnvironment