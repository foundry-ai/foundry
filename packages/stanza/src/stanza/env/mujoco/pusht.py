from stanza.env import (
    EnvWrapper, RenderConfig, 
    ImageRender, SequenceRender,
    HtmlRender, Environment,
    EnvironmentRegistry
)
from stanza.policy.transforms import Transform, chain_transforms
from stanza.policy import Policy
from stanza.dataclasses import dataclass, field
import dataclasses
from stanza import canvas
from stanza.env.mujoco import MujocoEnvironment, MujocoState

import shapely.geometry as sg

from functools import partial, cached_property

from mujoco.mjx._src.forward import fwd_position, fwd_velocity
from mujoco import mjx
import mujoco
import numpy as np
import jax.numpy as jnp
import jax.random

GOAL_POS = jnp.array([0, 0])
GOAL_ROT = jnp.array(-jnp.pi/4)

AGENT_RADIUS=15/252
BLOCK_SCALE=30/252
COM = 0.5*(BLOCK_SCALE/2) + 0.5*(2.5*BLOCK_SCALE)
XML = f"""
<mujoco>
<option timestep="0.05"/>
<worldbody>
    # The manipulator agent body
    <body pos="0.5 0.5 0" name="agent">
        # TODO: Replace with cylinder when MJX supports
        <geom type="sphere" size="{AGENT_RADIUS:.4}" pos="0 0 {AGENT_RADIUS:.4}" mass="0.1" rgba="0.1 0.1 0.9 1"/>
        <joint type="slide" axis="1 0 0" damping="0.1" stiffness="0" ref="0.5" name="agent_x"/>
        <joint type="slide" axis="0 1 0" damping="0.1" stiffness="0" ref="0.5" name="agent_y"/>
    </body>
    # The block body
    <body pos="-0.5 -0.5 0" name="block">
        # The horizontal box
        <geom type="box" size="{2*BLOCK_SCALE:.4} {0.5*BLOCK_SCALE} 0.5" 
            pos="0 -{0.5*BLOCK_SCALE:.4} 0.5" mass="0.03" rgba="0.467 0.533 0.6 1"/>
        # The vertical box
        <geom type="box" size="{0.5*BLOCK_SCALE:.4} {1.5*BLOCK_SCALE:.4} 0.5"
            pos="0 -{2.5*BLOCK_SCALE} 0.5" mass="0.03" rgba="0.467 0.533 0.6 1"/>

        <joint type="slide" axis="1 0 0" damping="5" stiffness="0" ref="-0.5"/>
        <joint type="slide" axis="0 1 0" damping="5" stiffness="0" ref="-0.5"/>
        # Hinge through the block COM
        <joint type="hinge" axis="0 0 1" damping="0.3" stiffness="0" pos="0 {-COM:.4} 0"/>
    </body>
    # The boundary planes
    <geom pos="-1 0 0" size="2 2 0.1"  xyaxes="0 1 0 0 0 1" type="plane"/>
    <geom pos="1 0 0" size="2 2 0.1"   xyaxes="0 0 1 0 1 0" type="plane"/>
    <geom pos="0 -1 0" size="2 2 0.1"  xyaxes="0 0 1 1 0 0" type="plane"/>
    <geom pos="0 1 0" size="2 2 0.1"   xyaxes="1 0 0 0 0 1" type="plane"/>
</worldbody>
<actuator>
    <motor ctrllimited="true" ctrlrange="-10.0 10.0" gear="1.0" joint="agent_x"/>
    <motor ctrllimited="true" ctrlrange="-10.0 10.0" gear="1.0" joint="agent_y"/>
</actuator>
</mujoco>
"""

@dataclass
class PushTObs:
    agent_pos: jnp.array
    agent_vel: jnp.array

    block_pos: jnp.array
    block_vel: jnp.array

    block_rot: jnp.array
    block_rot_vel: jnp.array

@dataclass
class PushTPosObs:
    agent_pos: jnp.array
    block_pos: jnp.array
    block_rot: jnp.array

@dataclass
class PushTKeypointObs:
    agent_pos: jnp.array
    block_pos: jnp.array
    block_end: jnp.array

@dataclass
class PushTKeypointRelObs:
    agent_block_pos: jnp.array
    agent_block_end: jnp.array
    rel_block_pos: jnp.array
    rel_block_end: jnp.array


@dataclass
class PushTEnv(MujocoEnvironment):
    success_threshold: float = 0.9

    @jax.jit
    def sample_action(self, rng_key: jax.Array):
        pos_agent = jax.random.randint(rng_key, (2,), 50, 450).astype(float)
        return pos_agent

    @jax.jit
    def sample_state(self, rng_key):
        return self.reset(rng_key)

    @jax.jit
    def reset(self, rng_key):
        a_pos, b_pos, b_rot, c = jax.random.split(rng_key, 4)
        agent_pos = jax.random.uniform(a_pos, (2,), minval=-0.8, maxval=0.8)
        block_rot = jax.random.uniform(b_pos, (), minval=-jnp.pi, maxval=jnp.pi)
        block_pos = jax.random.uniform(b_rot, (2,), minval=-0.4, maxval=0.4)
        # re-generate block positions
        # while the block is too close to the agent
        # (0.05 + 0.2)*sqrt(2) = 0.36 (approx)
        def gen_pos(carry):
            rng_key, _ = carry
            rng_key, sk = jax.random.split(rng_key)
            return (rng_key, jax.random.uniform(sk, (2,), minval=-0.4, maxval=0.4))
        _, block_pos = jax.lax.while_loop(
            lambda s: jnp.linalg.norm(s[1] - agent_pos) < 0.36,
            gen_pos, (c, block_pos)
        )
        qpos = jnp.concatenate([agent_pos, block_pos, block_rot[jnp.newaxis]])
        return MujocoState(
            qpos,
            jnp.zeros_like(qpos)
        )
    
    # For computing the reward

    @staticmethod
    def _block_points(pos, rot):
        center_a, hs_a = jnp.array([0, -BLOCK_SCALE/2]), \
                        jnp.array([2*BLOCK_SCALE, BLOCK_SCALE/2])
        center_b, hs_b = jnp.array([0, -2.5*BLOCK_SCALE]), \
                        jnp.array([BLOCK_SCALE/2, 1.5*BLOCK_SCALE])
        points = jnp.array([
            center_a + jnp.array([hs_a[0], -hs_a[1]]),
            center_a + hs_a,
            center_a + jnp.array([-hs_a[0], hs_a[1]]),
            center_a - hs_a,
            center_b + jnp.array([-hs_b[0], hs_b[1]]),
            center_b - hs_b,
            center_b + jnp.array([hs_b[0], -hs_b[1]]),
            center_b + hs_b
        ])
        rotM = jnp.array([
            [jnp.cos(rot), -jnp.sin(rot)],
            [jnp.sin(rot), jnp.cos(rot)]
        ])
        points = jax.vmap(lambda v: rotM @ v)(points)
        # points = jax.vmap(lambda v: rotM @ (v - com) + com)(points)
        return points + pos

    @staticmethod
    def _overlap(pointsA, pointsB):
        polyA = sg.Polygon(pointsA)
        polyB = sg.Polygon(pointsB)
        return polyA.intersection(polyB).area / polyA.area

    @jax.jit
    def reward(self, state, action, next_state):
        obs = self.observe(next_state)
        goal_points = self._block_points(GOAL_POS, GOAL_ROT)
        points = self._block_points(obs.block_pos, obs.block_rot)
        return jax.pure_callback(
            PushTEnv._overlap,
            jax.ShapeDtypeStruct((), jnp.float32),
            goal_points, points
        )
    
    @jax.jit
    def observe(self, state):
        mjx_data = mjx.make_data(self.mjx_model)
        mjx_data = mjx_data.replace(qpos=state.q, qvel=state.qd)
        mjx_data = mjx.forward(self.mjx_model, mjx_data)
        return PushTObs(
            # Extract agent pos, vel
            agent_pos=mjx_data.xpos[1,:2],
            agent_vel=mjx_data.cvel[1,3:5],
            # Extract block pos, vel, angle, angular vel
            block_pos=mjx_data.xpos[2,:2],
            block_rot=MujocoEnvironment._quat_to_angle(mjx_data.xquat[2,:4]),
            block_vel=mjx_data.cvel[2,3:5],
            block_rot_vel=mjx_data.cvel[2,2],
        )

    @staticmethod
    def _render_block(pos, rot, color):
        com = jnp.array([0, -0.15])
        # geom = canvas.fill(canvas.union(
        #     canvas.box((-0.2, 0), (0.2, 0.1)),
        #     canvas.box((-0.05, 0.1), (0.05, 0.4))
        # ), color=color)
        geom = canvas.fill(canvas.union(
            canvas.box((-2*BLOCK_SCALE, 0), (2*BLOCK_SCALE, BLOCK_SCALE)),
            canvas.box((-BLOCK_SCALE/2, BLOCK_SCALE), (BLOCK_SCALE/2, 4*BLOCK_SCALE))
        ), color=color)
        return canvas.transform(geom,
            #canvas.transform(geom, translation=-com),
            translation=pos*jnp.array([1, -1]),#com + pos*jnp.array([1, -1]),
            rotation=rot
        )

    @staticmethod
    def _render_agent(pos, color):
        return canvas.fill(
            canvas.circle(pos * jnp.array([1, -1]), AGENT_RADIUS),
            color=color
        )

    @partial(jax.jit, static_argnums=(2,3))
    def _render_image(self, obs : PushTObs, width : int, height : int):
        image = 0.95*jnp.ones((width, height, 3))
        # render just the block at its target position
        goal_t = PushTEnv._render_block(
            GOAL_POS, GOAL_ROT,
            (0.1, 0.8, 0.1)
        )
        curr_t = PushTEnv._render_block(
            obs.block_pos, obs.block_rot,
            canvas.colors.LightSlateGray
        )
        agent = PushTEnv._render_agent(
            obs.agent_pos, canvas.colors.Blue
        )
        world = canvas.stack(goal_t, curr_t, agent)
        translation = (1, 1)
        scale = (width/2, height/2)
        world = canvas.transform(world,
            translation=translation,
            scale=scale
        )
        image = canvas.paint(image, world)
        return image

    @jax.jit
    def render(self, config: RenderConfig, state: MujocoState) -> jax.Array:
        if type(config) == ImageRender:
            obs = PushTEnv.observe(self, state)
            return self._render_image(obs, config.width, config.height)
        if type(config) == SequenceRender:
            obs = PushTEnv.observe(self, state)
            return self._render_image(obs, config.width, config.height)
        elif type(config) == HtmlRender:
            if state.q.ndim == 1:
                state = jax.tree_map(lambda x: x[None], state)
            return MujocoEnvironment.brax_render(self.mj_model, state)


        
# A state-feedback adapter for the PushT environment
# Will run a PID controller under the hood
@dataclass
class PositionalControlTransform(Transform):
    k_p : float = 15
    k_v : float = 2

    def transform_policy(self, policy):
        return PositionalControlPolicy(policy, self.k_p, self.k_v)
    
    def transform_env(self, env):
        return PositionalControlEnv(env, self.k_p, self.k_v)

@dataclass
class PositionalControlPolicy(Policy):
    policy: Policy
    k_p : float = 20
    k_v : float = 2

    @property
    def rollout_length(self):
        return self.policy.rollout_length

    def __call__(self, input):
        output = self.policy(input)
        if output.action is None:
            return dataclasses.replace(output, action=jnp.zeros((2,)))
        a = self.k_p * (output.action - output.agent.position) + self.k_v * (-output.agent.velocity)
        return dataclasses.replace(
            output, action=a
        )

@dataclass
class PositionalControlEnv(EnvWrapper):
    k_p : float = 50
    k_v : float = 2

    def step(self, state, action, rng_key=None):
        obs = PushTEnv.observe(self.base, state)
        if action is not None:
            a = self.k_p * (action - obs.agent_pos) + self.k_v * (-obs.agent_vel)
        else: 
            a = jnp.zeros((2,))
        return self.base.step(state, a, None)
        # def step_fn(_, state):
        #     obs = PushTEnv.observe(self.base, state)
        #     if action is not None:
        #         a = self.k_p * (action - obs.agent_pos) + self.k_v * (-obs.agent_vel)
        #     else: 
        #         a = jnp.zeros((2,))
        #     return self.base.step(state, a, None)
        # return jax.lax.fori_loop(0, 6, step_fn, state)

@dataclass
class PositionalObsTransform(Transform):
    def transform_policy(self, policy):
        return PositionalObsPolicy(policy)
    
    def transform_env(self, env):
        return PositionalObsEnv(env)

@dataclass
class PositionalObsPolicy(Policy):
    policy: Policy

    @property
    def rollout_length(self):
        return self.policy.rollout_length

    def __call__(self, input):
        obs = input.observation
        obs = PushTPosObs(
            agent_pos=obs.agent.position,
            block_pos=obs.block.position,
            block_rot=obs.block.rotation
        )
        input = dataclasses.replace(input, observation=obs)
        return self.policy(input)

@dataclass
class PositionalObsEnv(EnvWrapper):
    def observe(self, state):
        obs = self.base.observe(state)
        return PushTPosObs(
            agent_pos=obs.agent_pos,
            block_pos=obs.block_pos,
            block_rot=obs.block_rot
        )

@dataclass
class KeypointObsTransform(Transform):
    def transform_policy(self, policy):
        raise NotImplementedError()
    
    def transform_env(self, env):
        return KeypointObsEnv(env)

@dataclass
class KeypointObsEnv(EnvWrapper):
    def observe(self, state):
        obs = self.base.observe(state)
        rotM = jnp.array([
            [jnp.cos(obs.block_rot), -jnp.sin(obs.block_rot)],
            [jnp.sin(obs.block_rot), jnp.cos(obs.block_rot)]
        ])
        end = rotM @ jnp.array([0, -4*BLOCK_SCALE]) + obs.block_pos
        return PushTKeypointObs(
            agent_pos=obs.agent_pos,
            block_pos=obs.block_pos,
            block_end=end
        )

@dataclass
class RelKeypointTransform(Transform):
    def transform_policy(self, policy):
        raise NotImplementedError()
    
    def transform_env(self, env):
        return RelKeypointEnv(env)

@dataclass
class RelKeypointEnv(EnvWrapper):
    def observe(self, state):
        obs = self.base.observe(state)
        rotM = jnp.array([
            [jnp.cos(obs.block_rot), -jnp.sin(obs.block_rot)],
            [jnp.sin(obs.block_rot), jnp.cos(obs.block_rot)]
        ])
        end = rotM @ jnp.array([0, -4*BLOCK_SCALE]) + obs.block_pos

        rotM = jnp.array([
            [jnp.cos(GOAL_ROT), -jnp.sin(GOAL_ROT)],
            [jnp.sin(GOAL_ROT), jnp.cos(GOAL_ROT)]
        ])
        goal_end = rotM @ jnp.array([0, -4*BLOCK_SCALE]) + GOAL_POS
        return PushTKeypointRelObs(
            agent_block_pos=obs.agent_pos - obs.block_pos,
            agent_block_end=obs.agent_pos - end,
            rel_block_pos=obs.block_pos - GOAL_POS,
            rel_block_end=end - goal_end,
        )

    def step(self, state, action, rng_key=None):
        # obs = self.base.observe(state)
        # action = action + obs.block_pos
        res = self.base.step(state, action, rng_key)
        return res

environments = EnvironmentRegistry[PushTEnv]()
environments.register("", PushTEnv)
def _make_positional(**kwargs):
    env = PushTEnv(**kwargs)
    return chain_transforms(
        PositionalControlTransform(),
        PositionalObsTransform
    ).transform_env(env)
environments.register("positional", _make_positional)