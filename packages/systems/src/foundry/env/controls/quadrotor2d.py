from typing import NamedTuple

import foundry.numpy as jnp
import jax
import plotly.express as px

import foundry.canvas as canvas

from foundry.env import Environment, EnvironmentRegistry

class State(NamedTuple):
    x: jax.Array
    z: jax.Array
    phi: jax.Array
    x_dot: jax.Array
    z_dot: jax.Array
    phi_dot: jax.Array

class Action(NamedTuple):
    thrust: jax.Array
    torque: jax.Array

class QuadrotorEnv(Environment[State, jax.Array, State]):
    def __init__(self):
        self.g = 9.8
        self.m = 0.8
        self.L = 0.086
        self.Ixx = 0.5
        self.dt = 0.05

    def sample_action(self, rng_key : jax.Array) -> Action:
        a, b = jax.random.split(rng_key)
        thrust = jax.random.uniform(a, (), jnp.float32, -1, 1)
        torque = jax.random.uniform(b, (), jnp.float32, -1, 1)
        return Action(thrust, torque)

    def sample_state(self, rng_key : jax.Array) -> State:
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
    
    def reset(self, rng_key : jax.Array) -> State:
        x_key, xd_key, z_key, zd_key, p_key, pd_key = jax.random.split(rng_key, 6)
        return State(
            x=jax.random.uniform(x_key, (), jnp.float32, -2., 2.),
            z=jax.random.uniform(z_key, (), jnp.float32, -2., 2.),
            phi=jax.random.uniform(p_key, (), jnp.float32, -1, 1),
            x_dot=jax.random.uniform(xd_key, (), jnp.float32, -3, 3),
            z_dot=jax.random.uniform(zd_key, (), jnp.float32, -3, 3),
            phi_dot=jax.random.uniform(pd_key, (), jnp.float32, -0.3, 0.3)
        )

    def step(self, state : State, action : Action, rng_key=None) -> State:
        thrust = action.thrust
        torque = action.torque

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
                state.phi**2 + \
                (state.x_dot**2 + state.z_dot**2 + \
                state.phi_dot**2)
        u_cost = jnp.mean((action.thrust - self.g*self.m)**2 + 0.2*action.torque**2)
        cost = jnp.mean(x_cost) + 0.5*u_cost
        return cost
    
    def reward(self, state, action, next_state):
        x_cost = state.x**2 + state.z**2 + \
                state.phi**2 + \
                (state.x_dot**2 + state.z_dot**2 + \
                state.phi_dot**2)
        u_cost = jnp.mean((action.thrust - self.g*self.m)**2 + 0.2*action.torque**2)
        cost = jnp.mean(x_cost) + 0.5*u_cost
        rew = -cost
        return jnp.exp(rew)
    
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
            traj = canvas.batch_union(canvas.circle(xy/3, 0.01*jnp.ones((xy.shape[0]))))
            objects.append(canvas.fill(traj, (0.1, 0.1, 0.6)))
        sdf = canvas.transform(
            canvas.stack(*objects),
            scale=jnp.array([width/1.5, height/1.5])
        )
        sdf = canvas.transform(
            sdf,
            translation=jnp.array([width/2, height/2])
        )
        image = canvas.paint(image, sdf) 
        return image

    def render(self, state, *, width=256, height=256, mode="image",
                                    state_trajectory=None, **kwargs):
        if mode == "image":
            return self._render_image(state, width=width, height=height,
                                      state_trajectory=state_trajectory)

environments = EnvironmentRegistry[QuadrotorEnv]()
environments.register("", QuadrotorEnv)