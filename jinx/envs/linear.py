from jinx.envs import Environment

import jax.numpy as jnp
from typing import NamedTuple

class State(NamedTuple):
    x: jnp.ndarray

class LinearSystem(Environment):
    def __init__(self, A, B, Q, R, Q_F):
        self.A = A
        self.B = B

        self.Q = Q
        self.R = R
        self.Q_F = Q_F

    @property
    def action_size(self):
        return 1
    
    def reset(self, key):
        return State(jnp.zeros((self.A.shape[-1])))
    
    def step(self, state, action):
        x = self.A @ state.x + self.B @ action
        return State(x)
    
    def cost(self, xs, us):
        xs = jnp.expand_dims(xs,-1)
        us = jnp.expand_dims(us,-1)
        x_cost = xs.T @ self.Q @ xs
        u_cost = us.T @ self.R @ us
        x_f_cost = xs[-1].T @ self.Q @ xs[-1]
        # add all the cost terms together
        return jnp.sum(x_cost) + jnp.sum(u_cost) + jnp.sum(x_f_cost)
    
    def render(self, state, width=256, height=256):
        return None