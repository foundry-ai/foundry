class State(NamedTuple):
    phi: jnp.ndarray
    theta: jnp.ndarray
    psi: jnp.ndarray
    p: jnp.ndarray
    q: jnp.ndarray
    r: jnp.ndarray

    x: jnp.ndarray
    y: jnp.ndarray
    z: jnp.ndarray
    u: jnp.ndarray
    v: jnp.ndarray
    w: jnp.ndarray

class QuadrotorEnvironment(Environment):
    def __init__(self):
        self.g = 9.81
        self.m = 0.468
        self.l = 0.225
        self.k = 2.98e-6
        self.b = 1.140e-7
        self.I_m = 3.357e-5

        self.I_xx = 4.856e-3
        self.I_yy = 4.856e-3
        self.I_zz = 8.801e-3

        self.A_x = 0.25
        self.A_y = 0.25
        self.A_z = 0.25
    
    def sample_action(self, rng_key):
        pass
    
    def step(state, action):
        tan_theta = jnp.tan(state.theta)
        sin_theta = jnp.sin(state.theta)
        cos_theta = jnp.cos(state.theta)
        sin_phi = jnp.sin(state.phi)
        cos_phi = jnp.cos(state.phi)
        sin_psi = jnp.sin(state.psi)
        cos_psi = jnp.cos(state.psi)

        phi_dot = state.p + tan_theta*(state.r*cos_psi+ state.q*sin_psi)
        theta_dot = state.q*cos_psi - state.r*sin_psi
        psi_dot = state.r*cos_phi/cos_theta + state.q*sin_psi/cos_theta

        p_dot = 