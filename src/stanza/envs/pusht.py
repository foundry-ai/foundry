from stanza.envs import (
    Wrapper, RenderConfig, 
    ImageRender, SequenceRender,
    HtmlRender, Environment
)
from stanza.policy.transforms import Transform
from stanza.policy import Policy
from stanza.util import AttrMap

from stanza import canvas, struct

import stanza.envs.planar as planar

from functools import partial, cached_property

from mujoco.mjx._src.forward import fwd_position, fwd_velocity
from mujoco import mjx
import mujoco
import numpy as np
import jax.numpy as jnp
import jax.random

def builder() -> planar.WorldBuilder:
    scale = 0.1
    world_scale = 1
    block_length = 4
    agent_radius = 0.5
    builder = planar.WorldBuilder(world_scale, world_scale, 0.005)
    builder.add_body(planar.Body(name="agent", geom=[
        planar.Circle(
            radius=scale*agent_radius, 
            mass=0.1, color=(0.1, 0.1, 0.9)
        )], pos=(0.5, 0.5), hinge=False,
        vel_damping=0.1
    ))
    builder.add_body(planar.Body(name="block", geom=[
        planar.Box(
            half_size=(scale*block_length/2, scale/2),
            mass=0.05, pos=(0., -scale/2),
            color=canvas.colors.LightSlateGray
        ),
        planar.Box(
            half_size=(scale/2, scale*(block_length - 1)/2),
            mass=0.05, pos=(0., -scale-scale*(block_length - 1)/2),
            color=canvas.colors.LightSlateGray
        )], pos=(-0.5, -0.5), vel_damping=5, rot_damping=0.5)
    )
    return builder
WORLD = builder()
MODEL = WORLD.load_model()

@struct.dataclass
class PushTObservation:
    agent_pos: jnp.array
    agent_vel: jnp.array

    block_pos: jnp.array
    block_vel: jnp.array

    block_rot: jnp.array
    block_rot_vel: jnp.array

@struct.dataclass
class PushTPosObs:
    agent_pos: jnp.array
    block_pos: jnp.array
    block_rot: jnp.array

@struct.dataclass
class PushTState:
    q: jax.Array
    qd: jax.Array

@struct.dataclass
class PushTEnv(Environment):
    success_threshold: float = 0.9

    goal_pos : jax.Array = jnp.array([-0.3, -0.3])
    goal_rot : jax.Array = jnp.array(jnp.pi/4)

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
        assert qpos.shape == MODEL.qpos0.shape
        return PushTState(
            qpos,
            jnp.zeros_like(MODEL.qpos0)
        )
    
    @jax.jit
    def reward(self, state):
        return jax.pure_callback(
            PushTEnv._callback_reward,
            jax.ShapeDtypeStruct((), jnp.float32),
            self, state
        )
    
    @jax.jit
    def step(self, state, action, rng_key=None):
        data = mjx.make_data(MODEL)
        data = data.replace(qpos=state.q, qvel=state.qd)
        if action is not None:
            xfrc_applied = data.xfrc_applied.at[1,:2].set(action)
            data = data.replace(xfrc_applied=xfrc_applied)
        @jax.jit
        def step_fn(_, data):
            return mjx.step(MODEL, data)
        data = jax.lax.fori_loop(0, 6, step_fn, data)
        return PushTState(data.qpos, data.qvel)
    
    @jax.jit
    def observe(self, state):
        mjx_data = mjx.make_data(MODEL)
        mjx_data = mjx_data.replace(qpos=state.q, qvel=state.qd)
        mjx_data = mjx.forward(MODEL, mjx_data)
        state = WORLD.extract_state(mjx_data)
        return PushTObservation(
            state['agent'].pos,
            state['agent'].vel,
            state['block'].pos,
            state['block'].vel,
            state['block'].rot,
            state['block'].rot_vel
        )
    
    @partial(jax.jit, static_argnums=(2,3))
    def _render_image(self, state, width, height):
        image = 0.95*jnp.ones((width, height, 3))
        # render just the block at its target position
        goal_state = planar.BodyState(pos=self.goal_pos, rot=self.goal_rot)
        builder = WORLD
        goal_t = builder.renderable({
            "block": goal_state
        }, {"block": (0.1, 0.8, 0.1)})
        world = builder.renderable(state)
        world = canvas.stack(goal_t, world)

        translation = (builder.world_half_x, builder.world_half_y)
        scale = (width/(2*builder.world_half_x),
                 height/(2*builder.world_half_y))
        world = canvas.transform(world,
            translation=translation,
            scale=scale
        )
        image = canvas.paint(image, world)
        return image

    @jax.jit
    def render(self, config: RenderConfig, state: PushTState) -> jax.Array:
        data = mjx.make_data(MODEL)
        data = data.replace(qpos=state.q, qvel=state.qd)
        data = mjx.forward(MODEL, data)
        if type(config) == ImageRender:
            state = WORLD.extract_state(data)
            return self._render_image(state, config.width, config.height)
        if type(config) == SequenceRender:
            state = WORLD.extract_state(data)
            return self._render_image(state, config.width, config.height)
        elif type(config) == HtmlRender:
            if data.qpos.ndim == 1:
                data = jax.tree_map(lambda x: x[None], data)
            return brax_render(WORLD.load_mj_model(), data)

def brax_to_state(sys, data):
    import brax.mjx.pipeline as pipeline
    from brax.base import Contact, Motion, System, Transform
    q, qd = data.qpos, data.qvel
    x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
    cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
    offset = data.xpos[1:, :] - data.subtree_com[sys.body_rootid[1:]]
    offset = Transform.create(pos=offset)
    xd = offset.vmap().do(cvel)
    data = pipeline._reformat_contact(sys, data)
    return pipeline.State(q=q, qd=qd, x=x, xd=xd, **data.__dict__)

def brax_render(mj_model, data_seq):
    import brax
    import brax.io.mjcf
    import brax.io.html
    sys = brax.io.mjcf.load_model(mj_model)
    T = data_seq.xpos.shape[0]
    states = jax.vmap(brax_to_state, in_axes=(None, 0))(sys, data_seq)
    states = [jax.tree_map(lambda x: x[i], states) for i in range(T)]
    return brax.io.html.render(sys, states)
        
# A state-feedback adapter for the PushT environment
# Will run a PID controller under the hood
@struct.dataclass
class PositionControlTransform(Transform):
    k_p : float = 15
    k_v : float = 2

    def transform_policy(self, policy):
        return PositionControlPolicy(policy, self.k_p, self.k_v)
    
    def transform_env(self, env):
        return PositionalControlEnv(env, self.k_p, self.k_v)

@struct.dataclass
class PositionControlPolicy(Policy):
    policy: Policy
    k_p : float = 20
    k_v : float = 2

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
        input = struct.replace(input, observation=obs)
        output = self.policy(input)
        a = self.k_p * (output.action - obs.agent.position) + self.k_v * (-obs.agent.velocity)
        return struct.replace(
            output, action=a,
            info=AttrMap(output.info, target_pos=output.action)
        )

@struct.dataclass
class PositionalControlEnv(Wrapper):
    k_p : float = 10
    k_v : float = 2

    def observe(self, state):
        obs = self.base.observe(state)
        return PushTPosObs(
            agent_pos=obs.agent_pos,
            block_pos=obs.block_pos,
            block_rot=obs.block_rot
        )

    def step(self, state, action, rng_key=None):
        obs = PushTEnv.observe(self.base, state)
        a = self.k_p * (action - obs.agent_pos) + self.k_v * (-obs.agent_vel)
        res = self.base.step(state, a, rng_key)
        return res

@struct.dataclass
class PositionObsTransform(Transform):
    def transform_policy(self, policy):
        return PositionObsPolicy(policy)
    
    def transform_env(self, env):
        return PositionalObsEnv(env)

@struct.dataclass
class PositionObsPolicy(Policy):
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
        input = struct.replace(input, observation=obs)
        return self.policy(input)

@struct.dataclass
class PositionalObsEnv(Wrapper):
    def observe(self, state):
        obs = self.base.observe(state)
        return PushTPosObs(
            agent_pos=obs.agent_pos,
            block_pos=obs.block_pos,
            block_rot=obs.block_rot
        )