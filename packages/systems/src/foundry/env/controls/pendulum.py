from foundry.env import Environment, EnvironmentRegistry, EnvWrapper
from foundry.policy import PolicyOutput

import jax
import foundry.numpy as jnp
import math

from typing import NamedTuple
from foundry.core.dataclasses import dataclass, field
import foundry.canvas as canvas
from foundry.policy import PolicyInput
from foundry.policy.mpc import MPC

class PendulumState(NamedTuple):
    angle: jnp.ndarray
    vel: jnp.ndarray

@dataclass
class PendulumEnv(Environment):
    sub_steps : int = 1
    dt : float = 0.2
    target_goal : PendulumState = field(
        default_factory=lambda: PendulumState(
            angle=jnp.array([math.pi]), vel=jnp.array([0])
        )
    )

    def sample_action(self, rng_key):
        return jax.random.uniform(
            rng_key, shape=(1,), minval=-1.0, maxval=1.0)

    # Sample state should be like reset(),
    # but whereas reset() is meant to be a distribution
    # over initial conditions, sample_state() should
    # give a distribution with support over all possible (or reasonable) states
    def sample_state(self, rng_key):
        k1, k2 = jax.random.split(rng_key)
        angle = jax.random.uniform(k1, shape=(1,), 
                    minval=-2, maxval=2*math.pi + 2)
        vel = jax.random.uniform(k2, shape=(1,), minval=-2, maxval=2)
        return PendulumState(angle, vel)

    def reset(self, key):
        # pick random position between +/- radians from center
        angle = jax.random.uniform(key,shape=(1,), minval=-1,maxval=+1)
        vel = jnp.zeros((1,))
        return PendulumState(angle, vel)

    def step(self, state, action, rng_key):
        angle = state.angle + self.dt*state.vel
        vel = 0.99*state.vel - self.dt*self.dt/2*jnp.sin(state.angle) + self.dt*action
        state = PendulumState(angle, vel)
        return state
    
    # def observe(self, state):
    #     return jnp.stack((jnp.sin(state.angle), jnp.cos(state.angle), state.vel), -1)

    # If u is None, this is the terminal cost
    def cost(self, states, actions):
        end_state = self.target_goal
        x = jnp.stack((states.angle, states.vel), -1)
        end_x = jnp.stack((end_state.angle, end_state.vel), -1)
        # x = jnp.stack((jnp.sin(states.angle),
        #                jnp.cos(states.angle),
        #                states.vel), -1)
        # end_x = jnp.stack((jnp.sin(end_state.angle),
        #                    jnp.cos(end_state.angle),
        #                    end_state.vel), -1)
        diff = jnp.sum((x - end_x)**2, axis=-1)
        x_cost = jnp.sum(diff[:-1])
        xf_cost = diff[-1]
        u_cost = jnp.sum(actions**2)
        cost = 100*xf_cost + 2*x_cost + u_cost
        return cost[0]

    def reward(self, state, action, next_state):
        end_state = self.target_goal
        pos = jnp.array([jnp.sin(state.angle), jnp.cos(state.angle)])
        end_pos = jnp.array([jnp.sin(end_state.angle), jnp.cos(end_state.angle)])
        angle_diff = jnp.sum((pos - end_pos)**2)
        vel_diff = (next_state.vel - end_state.vel)**2
        action_penalty = 0.5*jnp.sum(action**2)
        total = -angle_diff - vel_diff - action_penalty
        # reward of 1 == perfect
        return jnp.exp(10*total)/10

    def render(self, state):
        if mode == "image":
            image = jnp.ones((width, height, 3))
            pos = jnp.stack((jnp.sin(state.angle), jnp.cos(state.angle)))
            center = jnp.zeros((2,))
            circle_loc = center + pos
            stick = canvas.fill(canvas.segment(center, circle_loc, thickness=0.02), color=(0.,0.,0.))
            circle = canvas.fill(canvas.circle(circle_loc, radius=0.2), color=(1., 0., 0.))
            objects = [stick, circle]
            if state_trajectory is not None:
                pos = jnp.stack((jnp.sin(state_trajectory.angle), jnp.cos(state_trajectory.angle)), axis=-1)
                circles = canvas.batch_union(canvas.circle(pos, radius=0.02*jnp.ones((pos.shape[0],))))
                objects.append(canvas.fill(circles, color=(0., 0., 1.)))
            sdf = canvas.transform(canvas.stack(*objects), scale=width/3)
            sdf = canvas.transform(sdf, translation=(width/2, height/2))
            image = canvas.paint(image, sdf) 
            return image
        else:
            raise NotImplementedError("Mode not supported")
    
    def done(self, state):
        end_state = self.target_goal
        return jnp.logical_and(
            jnp.abs(state.angle - end_state.angle) < 0.03,
            jnp.abs(state.vel - end_state.vel) < 0.03
        )
    

environments = EnvironmentRegistry[PendulumEnv]()
environments.register("", PendulumEnv)