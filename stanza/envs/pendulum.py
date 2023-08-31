from stanza.envs import Environment
from stanza.policies import PolicyOutput

import jax
import jax.numpy as jnp
import math

from typing import NamedTuple
from functools import partial

from stanza.reporting import Figure, Video
from stanza.dataclasses import dataclass, field
import stanza.graphics.canvas as canvas

class State(NamedTuple):
    angle: jnp.ndarray
    vel: jnp.ndarray


# I added some weird  abstractions
# maybe ill remove them later
def get_goal(params):
    if params is None:
        return None
    else:
        return params.goal

@dataclass(jax=True)
class PendulumEnv(Environment):
    sub_steps : int = field(default=1, jax_static=True)
    dt : float = 0.2
    target_goal : State = State(angle=jnp.array(math.pi), vel=jnp.array(0))

    def sample_action(self, rng_key):
        return jax.random.uniform(
            rng_key, shape=(), minval=-1.0, maxval=1.0)

    # Sample state should be like reset(),
    # but whereas reset() is meant to be a distribution
    # over initial conditions, sample_state() should
    # give a distribution with support over all possible (or reasonable) states
    def sample_state(self, rng_key):
        k1, k2 = jax.random.split(rng_key)
        angle = jax.random.uniform(k1, shape=(), 
                    minval=-2, maxval=math.pi + 1)
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
        end_state = self.target_goal
        x = jnp.stack((x.angle, x.vel), -1)
        diff = (x - jnp.array([end_state.angle, end_state.vel]))
        x_cost = jnp.sum(diff[:-1]**2)
        xf_cost = jnp.sum(diff[-1]**2)
        if u == None:
            u_cost = 0
        else:
            u_cost = jnp.sum(u**2)
        return 5*xf_cost + 2*x_cost + u_cost

    def reward(self, state, action, next_state):
        end_state = self.target_goal
        angle_diff = next_state.angle - end_state.angle
        vel_diff = next_state.vel - end_state.vel
        angle_rew = -angle_diff**2
        vel_rew = -vel_diff**2
        action_penalty = 0.1 * jnp.sum(action**2)
        return angle_rew + vel_rew - action_penalty

    def render(self, state, width=256, height=256):
        angle = jnp.squeeze(state.angle)
        image = jnp.ones((width, height, 3))
        x, y = jnp.sin(angle), jnp.cos(angle)
        center = jnp.array([width/2, height/2])
        circle_loc = center + jnp.array([width*2/6, height*2/6])*jnp.stack((x,y))
        stick = canvas.segment(center, circle_loc, thickness=2)
        circle = canvas.circle(circle_loc, radius=width/24)
        sdf = canvas.stack(
            canvas.fill(stick, color=jnp.array([0.,0.,0.])),
            canvas.fill(circle, color=jnp.array([1.,0.,0.]))
        )
        image = canvas.paint(image, sdf) 
        return image
    
    def done(self, state):
        end_state = self.target_goal
        return jnp.logical_and(
            jnp.abs(state.angle - end_state.angle) < 0.03,
            jnp.abs(state.vel - end_state.vel) < 0.03
        )

    def teleop_policy(self, interface):
        def policy(_):
            left =  jnp.array(-0.1)*interface.key_pressed('a')
            right = jnp.array(0.1)*interface.key_pressed('d')
            return PolicyOutput(left + right)
        return policy

def builder(name):
    return PendulumEnv()