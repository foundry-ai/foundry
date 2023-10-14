from typing import NamedTuple

import jax.numpy as jnp
import jax
import plotly.express as px

import stanza.graphics.canvas as canvas
from stanza.envs import Environment

class State(NamedTuple):
    x: jnp.ndarray
    z: jnp.ndarray
    phi: jnp.ndarray
    x_dot: jnp.ndarray
    z_dot: jnp.ndarray
    phi_dot: jnp.ndarray

class QuadrotorEnvironment(Environment):
    def __init__(self):
        self.g = 0.1 # we made the gravity less terrible
                    # so that the cost for open loop
                    # would be less disastrous
        self.m = 0.8
        self.L = 0.086
        self.Ixx = 0.5
        self.dt = 0.1

    def sample_action(self, rng):
        return jax.random.uniform(rng, (2,), jnp.float32, -1, 1)

    def sample_state(self, rng_key):
        x_key, z_key, phi_key, xd_key, \
            zd_key, phid_key = jax.random.split(rng_key, 6)
        return State(
            x=jax.random.uniform(x_key, (), minval=-2, maxval=2),
            z=jax.random.uniform(z_key, (), minval=-2, maxval=2),
            phi=jax.random.uniform(phi_key, (), minval=-3, maxval=3),
            x_dot=jax.random.uniform(xd_key, (), minval=-2, maxval=2),
            z_dot=jax.random.uniform(zd_key, (), minval=-2, maxval=2),
            phi_dot=jax.random.uniform(phid_key, (), minval=-2, maxval=2)
        )
    
    def reset(self, rng_key):
        x_key, z_key = jax.random.split(rng_key, 2)
        return State(
            x=jax.random.uniform(x_key, (), jnp.float32, -1., 1.),
            z=jax.random.uniform(z_key, (), jnp.float32, -1., 1),
            phi=jnp.zeros(()),
            x_dot=jnp.zeros(()),
            z_dot=jnp.zeros(()),
            phi_dot=jnp.zeros(())
        )
    
    def step(self, state, action, rng_key):
        thrust = action[0]
        torque = action[1]

        x_dot = state.x_dot
        z_dot = state.z_dot
        phi_dot = state.phi_dot
        x_ddot = -thrust*jnp.sin(state.phi)/self.m
        z_ddot = thrust*jnp.cos(state.phi)/self.m - self.g
        phi_ddot = torque/self.Ixx

        dt = self.dt
        return State(
            x=state.x + dt*x_dot,
            z=state.z + dt*z_dot,
            phi=state.phi + dt*phi_dot,
            x_dot=state.x_dot + dt*x_ddot,
            z_dot=state.z_dot + dt*z_ddot,
            phi_dot=state.phi_dot + dt*phi_ddot,
        )

    def cost(self, state, action):
        x_cost = state.x**2 + state.z**2 + \
                5*state.phi**2 + \
                0.1*(state.x_dot**2 + state.z_dot**2 + \
                state.phi_dot**2)
        u_cost = jnp.mean(action**2)
        cost = jnp.mean(x_cost) + 0.1*u_cost + x_cost[-1]
        return cost
    
    def reward(self, state, action, next_state):
        x_cost = next_state.x**2 + next_state.z**2 + \
                5*next_state.phi**2 + \
                0.1*(next_state.x_dot**2 + next_state.z_dot**2 + \
                next_state.phi_dot**2)
        u_cost = jnp.mean(action**2)
        rew = -x_cost - u_cost
        return jnp.exp(10*rew)/10
    
    def _render_image(self, state : State, *, width=256, height=256,
                        state_trajectory: State = None):
        image = jnp.ones((width, height, 3))
        quad_width = 0.3
        quad_height = 0.05
        frame = canvas.rectangle(
            jnp.array([-quad_width/2, -quad_height/2]),
            jnp.array([quad_width/2, quad_height/2])
        )
        motor = canvas.stack(
            canvas.fill(canvas.rectangle([-quad_width/20,-0.5*quad_height],
                    [quad_width/20, quad_height*0.75]), 
                    color=jnp.array([0.3,0.3,0.3])),
            canvas.fill(canvas.rectangle([-quad_width/20,-quad_height],
                    [quad_width/20, -quad_height*0.5]), 
                    color=jnp.array([0.9,0.3,0.3])),
        )
        quadrotor = canvas.stack(
            canvas.fill(frame, color=jnp.array([0.1,0.1,0.1])),
            canvas.transform(motor, translation=jnp.array([-quad_width/2,0.])),
            canvas.transform(motor, translation=jnp.array([quad_width/2,0.])),
        )
        quadrotor = canvas.transform(
            quadrotor,
            translation=jnp.array([state.x/3, state.z/3]),
            rotation=state.phi
        )
        objects = [quadrotor]
        if state_trajectory is not None:
            xy = jnp.stack([state_trajectory.x, state_trajectory.z], axis=1)
            traj = canvas.vmap_union(canvas.circle(xy/3, 0.01*jnp.ones((xy.shape[0]))))
            objects.append(canvas.fill(traj, (0.1, 0.1, 0.6)))
        sdf = canvas.transform(
            canvas.stack(*objects),
            translation=jnp.array([1, 1]),
            scale=jnp.array([width/2, height/2])
        )
        image = canvas.paint(image, sdf) 
        return image

    def render(self, state, *, width=256, height=256, mode="image",
                                    state_trajectory=None, **kwargs):
        if mode == "image":
            return self._render_image(state, width=width, height=height,
                                      state_trajectory=state_trajectory)
    
    def visualize(self, states, actions):
        T = states.x.shape[0]
        x = px.line(x=jnp.arange(T), y=states.x)
        x.update_layout(xaxis_title="Time", yaxis_title="X", title="X")

        z = px.line(x=jnp.arange(T), y=states.z)
        z.update_layout(xaxis_title="Time", yaxis_title="Z", title="Z")

        phi = px.line(x=jnp.arange(T), y=states.phi)
        phi.update_layout(xaxis_title="Time", yaxis_title="Phi", title="Phi")

        pos = px.line(x=states.x, y=states.z)
        pos.update_layout(xaxis_title="X", yaxis_title="Z", title="Pos")

        thrust = px.line(x=jnp.arange(T-1), y=actions[:,0])
        thrust.update_layout(xaxis_title="Time", yaxis_title="Thrust", title="Thrust")
        torque = px.line(x=jnp.arange(T-1), y=actions[:,1])
        torque.update_layout(xaxis_title="Time", yaxis_title="Torque", title="Torque")
        return {
            'x': x,
            'z': z,
            'phi': phi,
            'pos': pos,
            'thrust': thrust,
            'torque': torque,
        }


def builder(name):
    return QuadrotorEnvironment()