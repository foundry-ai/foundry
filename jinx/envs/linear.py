from jinx.envs import Environment

import jax.numpy as jnp
import jax.random
from typing import NamedTuple
from functools import partial

class State(NamedTuple):
    x: jnp.ndarray

class LinearSystem(Environment):
    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B

        self.Q = Q
        self.R = R

    def sample_action(self, rng):
        return jax.random.uniform(rng, (1,), jnp.float32, -1, 1)
        #return jnp.zeros((1,))
    
    def reset(self, key):
        x = jax.random.uniform(key, (5,), jnp.float32, 8, 10)
        #x = jnp.zeros((5,))
        return State(x)
    
    def step(self, state, action):
        x = self.A @ jnp.expand_dims(state.x, -1) + self.B @ jnp.expand_dims(action, -1)
        return State(jnp.squeeze(x, -1))
    
    def cost(self, xs, us):
        x_cost = jnp.expand_dims(xs.x,-2) @ self.Q @ jnp.expand_dims(xs.x,-1)
        u_cost = jnp.expand_dims(us,-2) @ self.R @ jnp.expand_dims(us, -1)
        # add all the cost terms together
        return jnp.sum(x_cost) + jnp.sum(u_cost)
    
    def barrier(self, xs, us):
        constraints = [
                        jnp.ravel(us - 10),
                       jnp.ravel(-10 - us),
                       jnp.ravel(xs.x - 100),
                       jnp.ravel(-100 - xs.x)
                       ]
        return jnp.concatenate(constraints)
    
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
        Q=jnp.eye(5),
        R=jnp.eye(1)
    )