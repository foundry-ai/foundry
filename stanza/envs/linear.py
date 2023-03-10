from stanza.envs import Environment

import jax.numpy as jnp
import jax.random
from typing import NamedTuple
from functools import partial
import scipy

class LinearSystem(Environment):
    def __init__(self, A, B, Q, R,
                x0_min=None, x0_max=None,
                xmax=None, xmin=None,
                umax=None, umin=None):
        self.A = A
        self.B = B

        self.Q = Q
        self.R = R
        self.P = jnp.array(scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R,
                                              e=None, s=None, balanced=True))
        self.xmin = xmin
        self.xmax = xmax
        self.umin = umin
        self.umax = umax
        self.x0_min = x0_min if x0_min is not None else xmin
        self.x0_max = x0_max if x0_max is not None else xmax

    def sample_action(self, rng):
        min = self.umin if self.umin else -1
        max = self.umax if self.umax else 1
        return jax.random.uniform(rng, (self.B.shape[1],), jnp.float32, min, max)

    def sample_state(self, key):
        min = self.xmin if self.xmin else -1
        max = self.xmax if self.xmax else 1
        return jax.random.uniform(key, (self.A.shape[0],), jnp.float32, min, max)
    
    def reset(self, key):
        min = self.x0_min if self.x0_min else -1
        max = self.x0_max if self.x0_max else 1
        return jax.random.uniform(key, (self.A.shape[0],), jnp.float32, min, max)
    
    def step(self, state, action):
        action = jnp.atleast_1d(action)
        x = self.A @ jnp.expand_dims(state, -1) + self.B @ jnp.expand_dims(action, -1)
        return jnp.squeeze(x, -1)

    def cost(self, xs, us):
        x_cost = jnp.expand_dims(xs,-2) @ self.Q @ jnp.expand_dims(xs,-1)
        u_cost = jnp.expand_dims(us,-2) @ self.R @ jnp.expand_dims(us, -1)
        xf_cost = jnp.expand_dims(xs[-1],-2) @ self.P @ jnp.expand_dims(xs[-1],-1)
        return jnp.sum(x_cost) + jnp.sum(u_cost) + jnp.sum(xf_cost)
    
    def constraints(self, xs, us):
        constraints = []
        if self.umax:
            constraints.append(jnp.ravel(us - self.umax))
        if self.umin:
            constraints.append(jnp.ravel(self.umin - us))
        if self.xmax:
            constraints.append(jnp.ravel(xs - self.xmax))
        if self.xmin:
            constraints.append(jnp.ravel(self.xmin - xs))
        return jnp.concatenate(constraints) if constraints else jnp.zeros(())


def builder(name):
    env_path = name.split("/")
    sys = env_path[1] if len(env_path) > 1 else ""
    if sys == "di":
        return LinearSystem(
            A=jnp.array([
                [1., 1.],
                [0, 1.]
            ]),
            B=jnp.array([
                [0.],
                [1.],
            ]),
            Q=jnp.eye(2),
            R=0.01*jnp.eye(1),
            xmin=-10, xmax=10,
            x0_min=-8, x0_max=8,
            umin=-1, umax=1,
        )
    else:
        return LinearSystem( 
            A=jnp.array(
                [
                    [1.1, 0.860757747,  0.4110535,  0.17953273, -0.305308],
                    [0,   1.1,          0.4110535,  0.17953273, -0.305308],
                    [0  , 0,            1.1,        0.17953273, -0.305308],
                    [0  , 0,            0,          1.1,        -0.305308],
                    [0  , 0,            0,          0,          1.1],
                ]
            ),
            B=jnp.array([[0,0,0,0,1]]).T,
            Q=0.1*jnp.eye(5),
            R=0.1*jnp.eye(1)
        )