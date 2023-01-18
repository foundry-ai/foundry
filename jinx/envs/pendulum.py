from jinx.envs import Environment

import jax
import jax.numpy as jnp
import numpy as np
import math

from typing import NamedTuple
from cairo import ImageSurface, Context, Format
from functools import partial

import math

class State(NamedTuple):
    angle: jnp.ndarray
    vel: jnp.ndarray


class PendulumEnvironment(Environment):
    def __init__(self, sub_steps=1):
        self.sub_steps = sub_steps

    def sample_action(self, rng_key):
        return jax.random.uniform(rng_key, (1,), jnp.float32, -0.1, 0.1)

    # Sample state should be like reset(),
    # but whereas reset() is meant to be a distribution
    # over initial conditions, sample_state() should
    # give a distribution over reasonable states
    def sample_state(self, rng_key):
        k1, k2 = jax.random.split(rng_key)
        angle = jax.random.uniform(k1, shape=(1,), minval=-3.14,maxval=3.14)
        vel = jax.random.uniform(k2, shape=(1,), minval=-3,maxval=3)
        return State(angle, vel)

    def reset(self, key):
        # pick random position between +/- radians from center
        angle = jax.random.uniform(key,shape=(1,), minval=-1,maxval=1) + math.pi
        vel = jnp.zeros((1,))
        return State(angle, vel)

    def step(self, state, action):
        angle = state.angle + 0.1*state.vel
        vel = state.vel - 0.1*jnp.sin(state.angle + math.pi) + 0.1*action[0]
        state = State(angle, vel)
        return state
    
    # If u is None, this is the terminal cost
    def cost(self, x, u=None, t=None):
        x = jnp.concatenate((x.angle, x.vel), -1)
        diff = x - jnp.array([0, 0])
        if u is None:
            x_cost = jnp.sum(diff**2)
            return 10*x_cost
        else:
            x_cost = jnp.sum(diff**2)
            u_cost = jnp.sum(u**2)
            return x_cost + u_cost

    def barrier(self, _, us):
        constraints = [jnp.ravel(us - 3),
                       jnp.ravel(-3 - us)]
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
    theta = state.angle + math.pi

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