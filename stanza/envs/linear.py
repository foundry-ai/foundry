from stanza.envs import Environment

import jax.numpy as jnp
import jax.random
from typing import NamedTuple
from functools import partial
import scipy

class LinearSystem(Environment):
    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B

        self.Q = Q
        self.R = R
        self.P = jnp.array(scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R,
                                              e=None, s=None, balanced=True))

    def sample_action(self, rng):
        return jax.random.uniform(rng, (1,), jnp.float32, -1, 1)

    def sample_state(self, key):
        return jax.random.uniform(key, (5,), jnp.float32, -15, 15)
    
    def reset(self, key):
        return jax.random.uniform(key, (5,), jnp.float32, 8, 10)
    
    def step(self, state, action):
        x = self.A @ jnp.expand_dims(state, -1) + self.B @ jnp.expand_dims(action, -1)
        return jnp.squeeze(x, -1)
    
    def cost(self, x, u=None):
        if u is None:
            x_cost = jnp.expand_dims(x,-2) @ self.P @ jnp.expand_dims(x,-1)
            return jnp.sum(x_cost)
        else:
            x_cost = jnp.expand_dims(x,-2) @ self.Q @ jnp.expand_dims(x,-1)
            u_cost = jnp.expand_dims(u,-2) @ self.R @ jnp.expand_dims(u, -1)
            return jnp.sum(x_cost) + jnp.sum(u_cost)
    
    def barrier(self, xs, us):
        constraints = [
                       jnp.ravel(us - 10),
                       jnp.ravel(-10 - us),
                    #    jnp.ravel(xs.x - 100),
                    #    jnp.ravel(-100 - xs.x)
                    ]
        return jnp.concatenate(constraints)

    def barrier_feasible(self, x0, us):
        return jnp.zeros_like(us)
    
    def render(self, state, width=256, height=256):
        return None


def builder():
    return partial(LinearSystem, 
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