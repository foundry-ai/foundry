from typing import NamedTuple

import jax.numpy as jnp
import jax
import plotly.express as px

fromstanza.envs import Environment

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
        return jax.random.uniform(rng, (2,), jnp.float32, -1.5, 1.5)

    def sample_state(self, rng_key):
        x_key, z_key, phi_key, xd_key, \
            zd_key, phid_key = jax.random.split(rng_key, 6)
        return State(
            x=jax.random.uniform(x_key, (), minval=-3, maxval=3),
            z=jax.random.uniform(z_key, (), minval=-3, maxval=3),
            phi=jax.random.uniform(phi_key, (), minval=-3, maxval=3),
            x_dot=jax.random.uniform(xd_key, (), minval=-3, maxval=3),
            z_dot=jax.random.uniform(zd_key, (), minval=-3, maxval=3),
            phi_dot=jax.random.uniform(phid_key, (), minval=-3, maxval=3)
        )
    
    def reset(self, rng_key):
        x_key, z_key, phi_key, xd_key, \
            zd_key, phid_key = jax.random.split(rng_key, 6)
        return State(
            x=jax.random.uniform(x_key, (), jnp.float32, -0.5, 0.5),
            z=jax.random.uniform(z_key, (), jnp.float32, -0.5, 0.5),
            phi=jnp.zeros(()),
            x_dot=jnp.zeros(()),
            z_dot=jnp.zeros(()),
            phi_dot=jnp.zeros(())
        )
    
    def step(self, state, action):
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

    def cost(self, state, action=None, t=None):
        x_cost = state.x**2 + state.z**2 + \
                10*state.phi**2 + \
                0.1*(state.x_dot**2 + state.z_dot**2 + \
                state.phi_dot**2)
        cost = x_cost
        if action is not None:
            u_cost = 0.1*jnp.sum(action**2)
            cost = cost + u_cost
        else:
            cost = 10*cost
        return cost
    
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


def builder():
    return QuadrotorEnvironment