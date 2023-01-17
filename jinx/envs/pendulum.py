from jinx.envs import Environment

import jax
import jax.numpy as jnp
import numpy as np
import math

from typing import NamedTuple
from cairo import ImageSurface, Context, Format
from functools import partial

class State(NamedTuple):
    angle: jnp.ndarray
    vel: jnp.ndarray


class PendulumEnvironment(Environment):
    def __init__(self):
        pass
    
    def sample_action(self, rng_key):
        return jax.random.uniform(rng_key, (1,), jnp.float32, -0.1, 0.1)

    def reset(self, key):
        # pick random position between +/- radians from right
        angle = jax.random.uniform(key,shape=(1,), minval=-1,maxval=1)
        vel = jnp.zeros((1,))
        return State(angle, vel)

    def step(self, state, action):
        angle = state.angle + 0.05*state.vel
        vel = state.vel - 0.05*jnp.sin(state.angle) + 0.05*action[0]
        return State(angle, vel)
    
    # If u is None, this is the terminal cost
    def cost(self, x, u=None):
        x = jnp.concatenate((x.angle, x.vel), -1)
        diff = x - jnp.array([jnp.pi, 0])
        x_cost = jnp.sum(diff**2)
        if u is not None:
            u_cost = jnp.sum(u**2)
            return x_cost + 0.01*u_cost
        else:
            return 2*x_cost

    def barrier(self, _, us):
        constraints = [jnp.ravel(us - 1),
                       jnp.ravel(-1 - us)]
        return jnp.concatenate(constraints)

    def barrier_feasible(self, x0, us):
        return jnp.zeros_like(us)

    def render(self, state, width=256, height=256):
        return jax.pure_callback(
            partial(render_pendulum, width=width, height=height),
            jax.ShapeDtypeStruct((3, width, height), jnp.uint8),
            state
        )

def builder():
    return PendulumEnvironment

def render_pendulum(state, width, height):
    surface = ImageSurface(Format.ARGB32, width, height)
    ctx = Context(surface)
    ctx.rectangle(0, 0, width, height)
    ctx.set_source_rgb(0.9, 0.9, 0.9)
    ctx.fill()
    ctx.move_to(width/2, height/2)

    radius = 0.7*min(width, height)/2
    ball_radius = 0.1*min(width, height)/2

    # put theta through a tanh to prevent
    # wraparound
    theta = state.angle

    x = np.sin(theta)*radius + width/2
    y = np.cos(theta)*radius + height/2

    ctx.set_source_rgb(0.1, 0.1, 0.1)
    ctx.set_line_width(1)
    ctx.line_to(x, y)
    ctx.stroke()

    ctx.set_source_rgb(0.9, 0, 0)
    ctx.arc(x, y, ball_radius, 0, 2*math.pi)
    ctx.fill()
    img = cairo_to_numpy(surface)[:3,:,:]
    # we need to make a copy otherwise it will
    # get overridden the next time we render
    return np.copy(img)

def cairo_to_numpy(surface):
    data = np.ndarray(shape=(surface.get_height(), surface.get_width(), 4),
                    dtype=np.uint8,
                    buffer=surface.get_data())
    data[:,:,[0,1,2,3]] = data[:,:,[2,1,0,3]]
    data = np.transpose(data, (2, 0, 1))
    return data