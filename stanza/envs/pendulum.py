from stanza.envs import Environment

import jax
import jax.numpy as jnp
import numpy as np
import math

from typing import NamedTuple
from functools import partial
from stanza.runtime.database import Figure, Video

import math
import plotly.express as px

class State(NamedTuple):
    angle: jnp.ndarray
    vel: jnp.ndarray

class PendulumEnvironment(Environment):
    def __init__(self, sub_steps=1):
        self.sub_steps = sub_steps
        self.dt = 0.1

    def sample_action(self, rng_key):
        return jax.random.uniform(
            rng_key, shape=(), minval=-1.0, maxval=1.0)

    # Sample state should be like reset(),
    # but whereas reset() is meant to be a distribution
    # over initial conditions, sample_state() should
    # give a distribution with support over all possible (or reasonable) states
    def sample_state(self, rng_key):
        k1, k2 = jax.random.split(rng_key)
        angle = 5*jax.random.uniform(k1, shape=(), minval=-1, maxval=math.pi + 1)
        vel = 5*jax.random.uniform(k2, shape=(), minval=-1, maxval=1)
        return State(angle, vel)

    def reset(self, key):
        # pick random position between +/- radians from center
        angle = jax.random.uniform(key,shape=(), minval=math.pi-1,maxval=math.pi+1)
        vel = jnp.zeros(())
        return State(angle, vel)

    def step(self, state, action):
        angle = state.angle + self.dt*state.vel
        vel = state.vel - self.dt*jnp.sin(state.angle) + self.dt*action
        state = State(angle, vel)
        return state
    
    # If u is None, this is the terminal cost
    def cost(self, x, u=None):
        x = jnp.stack((x.angle, x.vel), -1)
        diff = (x - jnp.array([math.pi, 0]))**2
        x_cost = jnp.sum(diff)
        xf_cost = jnp.sum(diff[-1])
        u_cost = jnp.sum(u**2)
        return 100*xf_cost + 2*x_cost + u_cost

    def barrier(self, _, us):
        constraints = [jnp.ravel(us - 3),
                       jnp.ravel(-3 - us)]
        return jnp.concatenate(constraints)

    def barrier_feasible(self, x0, us):
        return jnp.zeros_like(us)
    
    def visualize(self, states, actions):
        traj = px.line(x=jnp.squeeze(states.angle, -1), y=jnp.squeeze(states.vel, -1))
        traj.update_layout(xaxis_title="Theta", yaxis_title="Omega", title="State Trajectory")

        theta = px.line(x=jnp.arange(states.angle.shape[0]), y=jnp.squeeze(states.angle, -1))
        theta.update_layout(xaxis_title="Time", yaxis_title="Theta", title="Angle")

        omega = px.line(x=jnp.arange(states.vel.shape[0]), y=jnp.squeeze(states.vel, -1))
        omega.update_layout(xaxis_title="Time", yaxis_title="Omega", title="Angular Velocity")

        u = px.line(x=jnp.arange(actions.shape[0]), y=jnp.squeeze(actions, -1))
        u.update_layout(xaxis_title="Time", yaxis_title="u")

        video = jax.vmap(self.render)(states)

        return {
            'video': Video(video, fps=15),
            'traj': Figure(traj),
            'theta': Figure(theta),
            'omega': Figure(omega),
            'u': Figure(u)
        }

    def render(self, state, width=256, height=256):
        raise NotImplementedError()
        # return jax.pure_callback(
        #     partial(render_pendulum, width=width, height=height),
        #     jax.ShapeDtypeStruct((3, width, height), jnp.uint8),
        #     state
        # )

def builder(name):
    return PendulumEnvironment()
