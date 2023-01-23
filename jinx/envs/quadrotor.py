class State(NamedTuple):
    x: jnp.ndarray
    x_dot: jnp.ndarray
    z: jnp.ndarray
    z_dot: jnp.ndarray
    phi: jnp.ndarray
    phi_dot: jnp.ndarray

class QuadrotorEnvironment(Environment):
    def __init__(self):
        self.g = 9.81
        self.m = 0.18
        self.L = 0.086
        self.Ixx = 0.00025
    
    def sample_action(self, rng_key):
        pass
    
    def step(state, action):
        pass