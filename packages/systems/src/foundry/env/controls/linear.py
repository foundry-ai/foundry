from foundry.env import Environment, EnvironmentRegistry

import foundry.numpy as jnp
import jax.random
from typing import NamedTuple
from functools import partial
import scipy
import jax

class LinearSystem(Environment):
    def __init__(self, A, B, Q=None, R=None,
                x0_min=None, x0_max=None,
                x_max=None, x_min=None,
                u_max=None, u_min=None):
        self.A = A
        self.B = B

        self.Q = Q if Q is not None else jnp.eye(A.shape[0])
        self.R = R if R is not None else jnp.eye(B.shape[1])
        self.P = jax.pure_callback(
            lambda A, B, Q, R: jnp.array(scipy.linalg.solve_discrete_are(A, B, Q, R,
                                              e=None, s=None, balanced=True), dtype=A.dtype),
            jax.ShapeDtypeStruct(A.shape, A.dytpe),
            self.A, self.B, self.Q, self.R
        )
        self.x_min = x_min
        self.x_max = x_max
        self.u_min = u_min
        self.u_max = u_max
        self.x0_min = x0_min if x0_min is not None else x_min
        self.x0_max = x0_max if x0_max is not None else x_max

    def sample_action(self, rng):
        min = self.u_min if self.u_min is not None else -1
        max = self.u_max if self.u_max is not None else 1
        return jax.random.uniform(rng, (self.B.shape[1],), jnp.float32, min, max)

    def sample_state(self, key):
        min = self.x_min if self.x_min is not None else -1
        max = self.x_max if self.x_max is not None else 1
        return jax.random.uniform(key, (self.A.shape[0],), jnp.float32, min, max)
    
    def reset(self, key):
        min = self.x0_min if self.x0_min is not None else -1
        max = self.x0_max if self.x0_max is not None else 1
        return jax.random.uniform(key, (self.A.shape[0],), jnp.float32, min, max)
    
    def step(self, state, action, rng_key = None):
        if action is None:
            x = self.A @ jnp.expand_dims(state, -1)
        else:
            action = jnp.atleast_1d(action)
            x = self.A @ jnp.expand_dims(state, -1) + self.B @ jnp.expand_dims(action, -1)
        return jnp.squeeze(x, -1)

    def cost(self, xs, us):
        x_cost = jnp.expand_dims(xs,-2) @ self.Q @ jnp.expand_dims(xs,-1)
        u_cost = jnp.expand_dims(us,-2) @ self.R @ jnp.expand_dims(us, -1)
        xf_cost = jnp.expand_dims(xs[-1],-2) @ self.P @ jnp.expand_dims(xs[-1],-1)
        return jnp.sum(x_cost) + jnp.sum(u_cost) + jnp.sum(xf_cost)

environments = EnvironmentRegistry[LinearSystem]()
environments.register("", LinearSystem)
environments.register("double_integrator", 
    partial(LinearSystem, 
        A=jnp.array([[1, 1], [0, 1]]),
        B=jnp.array([[0], [1]]),
        Q=jnp.array([[1, 0], [0, 1]]),
        R=0.01*jnp.array([[1]]),
        x0_min=jnp.array([-4, -4]),
        x0_max=jnp.array([4, 4]),
        x_min=jnp.array([-10, -10]),
        x_max=jnp.array([10, 10]),
        u_min=jnp.array([-1]),
        u_max=jnp.array([1])
    ))
environments.register("2d", 
    partial(LinearSystem, 
        A=jnp.array([[1.1, 1], [0, 1.1]]),
        B=jnp.array([[0], [1]]),
        Q=jnp.array([[1, 0], [0, 1]]),
        R=0.01*jnp.array([[1]]),
        x0_min=jnp.array([8, 8]),
        x0_max=jnp.array([10, 10]),
        x_min=jnp.array([-100]*2),
        x_max=jnp.array([100]*2),
        u_min=jnp.array([-10]),
        u_max=jnp.array([10]),
    ))
environments.register("3d",
    partial(LinearSystem,
        A = jnp.array([
            [1.1, 0.86075747, 0.4110535],
            [0., 1.1, 0.4110535],
            [0., 0., 1.1]
        ]),
        B=jnp.array([[0], [0], [1.]]),
        Q=jnp.eye(3),
        R=jnp.eye(1),
        x0_min=jnp.array([-8]*3),
        x0_max=jnp.array([8]*3),
        x_min=jnp.array([-100]*3),
        x_max=jnp.array([100]*3),
        u_min=jnp.array([-10]),
        u_max=jnp.array([10]),
    )
)

environments.register("4d",
    partial(LinearSystem,
        A=jnp.array(
            [[0.7, -0.1, 0.0, 0.0],
             [0.2, -0.5, 0.1, 0.0],
             [0.0, 0.1, 0.1, 0.0],
             [0.5, 0.0, 0.5, 0.5]]
        ),
        B=jnp.array(
            [[0.0, 0.1],
             [0.1, 1.0],
             [0.1, 0.0],
             [0.0, 0.0]]
        ),
        Q=jnp.eye(4),
        R=jnp.eye(2),
        # x0_min=jnp.array([8]*4), x0_max=jnp.array([10]*4),
        x_min=jnp.array([-100]*4), x_max=jnp.array([100]*4),
        u_min=jnp.array([-10]), u_max=jnp.array([10]),
    )
)
environments.register("5d",
    partial(LinearSystem,
        A=jnp.array([
            [1.1, 0.86075747, 0.4110535, 0.17953273, -0.3053808],
            [0., 1.1, 0.4110535, 0.17953273, -0.3053808],
            [0., 0., 1.1, 0.17953273, -0.3053808],
            [0., 0., 0., 1.1, -0.3053808],
            [0., 0., 0., 0., 1.1]
        ]),
        B=jnp.array([[0], [0], [0], [0], [1.]]),
        Q=jnp.eye(5),
        R=jnp.eye(1),
        x0_min=jnp.array([-8]*5),
        x0_max=jnp.array([8]*5),
        x_min=jnp.array([-100]*5),
        x_max=jnp.array([100]*5),
        u_min=jnp.array([-10]),
        u_max=jnp.array([10]),
    )
)