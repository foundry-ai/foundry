from stanza.envs import Environment
from stanza.policies import PolicyOutput

import jax
import jax.numpy as jnp
import numpy as np
import math

from typing import NamedTuple
from functools import partial
from stanza.runtime.database import Figure, Video
import stanza.graphics.canvas as canvas

import math

class State(NamedTuple):
    angle: jnp.ndarray
    vel: jnp.ndarray

class PendulumEnv(Environment):
    def __init__(self, sub_steps=1):
        self.sub_steps = sub_steps
        self.dt = 0.2

    def sample_action(self, rng_key):
        return jax.random.uniform(
            rng_key, shape=(), minval=-1.0, maxval=1.0)

    # Sample state should be like reset(),
    # but whereas reset() is meant to be a distribution
    # over initial conditions, sample_state() should
    # give a distribution with support over all possible (or reasonable) states
    def sample_state(self, rng_key):
        k1, k2 = jax.random.split(rng_key)
        angle = jax.random.uniform(k1, shape=(), minval=-2, maxval=math.pi + 1)
        vel = jax.random.uniform(k2, shape=(), minval=-1, maxval=1)
        return State(angle, vel)

    def reset(self, key):
        # pick random position between +/- radians from center
        angle = jax.random.uniform(key,shape=(), minval=-1,maxval=+1)
        vel = jnp.zeros(())
        return State(angle, vel)

    def step(self, state, action, rng_key):
        angle = state.angle + self.dt*state.vel
        vel = 0.99*state.vel - self.dt*self.dt/2*jnp.sin(state.angle) + self.dt*action
        state = State(angle, vel)
        return state
    
    # If u is None, this is the terminal cost
    def cost(self, x, u):
        x = jnp.stack((x.angle, x.vel), -1)
        diff = (x - jnp.array([math.pi, 0]))
        x_cost = jnp.sum(diff[:-1]**2)
        xf_cost = jnp.sum(diff[-1]**2)
        if u == None:
            u_cost = 0
        else:
            u_cost = jnp.sum(u**2)
        return 5*xf_cost + 2*x_cost + u_cost

    def constraints(self, _, us):
        constraints = [jnp.ravel(us - 3),
                       jnp.ravel(-3 - us)]
        return jnp.concatenate(constraints)

    def visualize(self, states, actions):
        import plotly.express as px
        traj = px.line(x=states.angle, y=states.vel)
        traj.update_layout(xaxis_title="Theta", yaxis_title="Omega", title="State Trajectory")

        theta = px.line(x=jnp.arange(states.angle.shape[0]), y=states.angle)
        theta.update_layout(xaxis_title="Time", yaxis_title="Theta", title="Angle")

        omega = px.line(x=jnp.arange(states.vel.shape[0]), y=states.vel)
        omega.update_layout(xaxis_title="Time", yaxis_title="Omega", title="Angular Velocity")

        u = px.line(x=jnp.arange(actions.shape[0]), y=actions)
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
        angle = jnp.squeeze(state.angle)
        image = jnp.ones((width, height, 3))
        x, y = jnp.sin(angle), jnp.cos(angle)
        center = jnp.array([width/2, height/2])
        circle_loc = center + jnp.array([width*2/6, height*2/6])*jnp.stack((x,y))
        stick = canvas.line(center, circle_loc)
        circle = canvas.circle(circle_loc, radius=width/24, color=jnp.array([1.,0.,0.]))
        sdf = canvas.stack(stick, circle)
        image = canvas.paint(image, sdf) 
        return image
    
    def teleop_policy(self, interface):
        def policy(_):
            left =  jnp.array(-0.1)*interface.key_pressed('a')
            right = jnp.array(0.1)*interface.key_pressed('d')
            return PolicyOutput(left + right)
        return policy

def builder(name):
    return PendulumEnv()