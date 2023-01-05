from jinx.envs import Environment

import jax.numpy as jnp
from typing import NamedTuple

class State(NamedTuple):
    x: jnp.ndarray

class LinearSystem(Environment):
    def __init__(self, A, B):
        self.A = A
        self.B = B