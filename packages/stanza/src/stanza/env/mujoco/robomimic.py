import jax
import jax.numpy as jnp

import mujoco
from mujoco import mjx
import robosuite

import dataclasses
from functools import partial

from stanza.dataclasses import dataclass, field
from stanza.env import (
    EnvWrapper, RenderConfig, 
    ImageRender, SequenceRender,
    HtmlRender, Environment,
    EnvironmentRegistry
)
from stanza.policy import Policy, PolicyInput
from stanza.policy.transforms import Transform, chain_transforms
from stanza.packages.stanza.src.stanza.env.mujoco.util import _quat_to_angle, brax_render
from stanza import canvas
from jax.random import PRNGKey

MJ_MODEL = None
MJX_MODEL = None
def load_mj_model():
    global MJ_MODEL
    if MJ_MODEL is None:
        MJ_MODEL = mujoco.MjModel.from_xml_string(XML)
    return MJ_MODEL

def load_mjx_model():
    global MJX_MODEL
    if MJX_MODEL is None:
        model = load_mj_model()
        with jax.ensure_compile_time_eval():
            MJX_MODEL = mjx.put_model(model)
    return MJX_MODEL

@dataclass
class ReacherObs:
    fingertip_pos: jnp.array
    fingertip_vel: jnp.array

    body0_pos: jnp.array
    body0_rot: jnp.array
    body0_rot_vel: jnp.array

    body1_pos: jnp.array
    body1_rot: jnp.array
    body1_rot_vel: jnp.array

@dataclass
class ReacherPosObs:
    fingertip_pos: jnp.array
    body0_pos: jnp.array
    body0_rot: jnp.array
    body1_pos: jnp.array
    body1_rot: jnp.array

@dataclass
class ReacherState:
    q: jax.Array
    qd: jax.Array

@dataclass
class ReacherEnv(Environment):

    target_id = load_mj_model().body("target").id
    body0_id = load_mj_model().body("body0").id
    body1_id = load_mj_model().body("body1").id
    fingertip_id = load_mj_model().body("fingertip").id
    target_pos = load_mj_model().body("target").pos[:2]

    @jax.jit
    def sample_action(self, rng_key: jax.Array):
        pos_agent = jax.random.randint(rng_key, (2,), 50, 450).astype(float)
        return pos_agent

    @jax.jit
    def sample_state(self, rng_key):
        return self.reset(rng_key)

    @jax.jit
    def reset(self, rng_key : jax.Array): 
        b0_rot, b1_rot = jax.random.split(rng_key, 2)
        body0_rot = jax.random.uniform(b0_rot, (), minval=-jnp.pi, maxval=jnp.pi)
        body1_rot = jax.random.uniform(b1_rot, (), minval=-jnp.pi, maxval=jnp.pi)

        qpos = jnp.concatenate([body0_rot[jnp.newaxis], body1_rot[jnp.newaxis], self.target_pos])
        return ReacherState(
            qpos,
            jnp.zeros_like(qpos)
        )

    @jax.jit
    def step(self, state, action, rng_key = None): 
        data = mjx.make_data(load_mjx_model())
        data = data.replace(qpos=state.q, qvel=state.qd)
        if action is not None:
            data = data.replace(ctrl=action)
        @jax.jit
        def step_fn(data, _):
            return mjx.step(load_mjx_model(), data), None
        data, _ = jax.lax.scan(step_fn, data, length=1)
        return ReacherState(data.qpos, data.qvel)
    
    @jax.jit
    def observe(self, state): 
        mjx_data = mjx.make_data(load_mjx_model())
        mjx_data = mjx_data.replace(qpos=state.q, qvel=state.qd)
        mjx_data = mjx.forward(load_mjx_model(), mjx_data)
        return ReacherObs(
            fingertip_pos=mjx_data.xpos[self.fingertip_id,:2],
            fingertip_vel=mjx_data.cvel[self.fingertip_id, 3:5],
            body0_pos=mjx_data.xpos[self.body0_id,:2],
            body0_rot=_quat_to_angle(mjx_data.xquat[self.body0_id,:4]),
            body0_rot_vel=mjx_data.cvel[self.body0_id, 2],
            body1_pos=mjx_data.xpos[self.body1_id,:2],
            body1_rot=_quat_to_angle(mjx_data.xquat[self.body1_id,:4]),
            body1_rot_vel=mjx_data.cvel[self.body1_id, 2]
        )
    
    @jax.jit
    def reward(self, state, action, next_state) -> jax.Array:
        obs = self.observe(next_state)
        reward_dist = -jnp.linalg.norm(obs.fingertip_pos - self.target_pos)
        reward_control = -jnp.sum(jnp.square(action))
        return reward_dist + reward_control

    

    @partial(jax.jit, static_argnums=(2,3))
    def _render_image(self, obs : ReacherObs, width : int, height : int):
        image = 0.95*jnp.ones((width, height, 3))
        target = canvas.fill(
            canvas.circle(self.target_pos * jnp.array([1, -1]), 0.009),
            color=canvas.colors.Red
        )
        body0 = canvas.transform(
            canvas.fill(
                canvas.box((0, -0.0055), (0.1, 0.0055)),
                color=canvas.colors.LightSlateGray
            ),
            translation=obs.body0_pos*jnp.array([1, -1]),
            rotation=obs.body0_rot
        )
        body1 = canvas.transform(
            canvas.fill(
                canvas.box((0, -0.0055), (0.1, 0.0055)),
                color=canvas.colors.Blue
            ),
            translation=obs.body1_pos*jnp.array([1, -1]),
            rotation=obs.body1_rot
        )
        world = canvas.stack(target, body0, body1)
        translation = (0.25,0.25)
        scale = (width/0.5, height/0.5)
        world = canvas.transform(world,
            translation=translation,
            scale=scale
        )
        image = canvas.paint(image, world)
        return image
    
    @jax.jit
    def render(self, config: RenderConfig, state: ReacherState) -> jax.Array:
        if type(config) == ImageRender or type(config) == SequenceRender:
            obs = ReacherEnv.observe(self, state)
            return self._render_image(obs, config.width, config.height)
        elif type(config) == HtmlRender:
            if data.qpos.ndim == 1:
                data = jax.tree_map(lambda x: x[None], data)
            return brax_render(load_mj_model(), data)