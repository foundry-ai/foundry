from stanza.envs import Environment, RenderConfig, ImageRender, HtmlRender
from stanza.policy.transforms import Transform
from stanza.policy import Policy
from stanza.util import AttrMap

from stanza import canvas, struct

import stanza.envs.planar as planar

from functools import partial, cached_property

import mujoco
from mujoco import mjx
import jax.numpy as jnp
import jax.random

@struct.dataclass
class PushTEnv(planar.PlanarEnvironment):
    success_threshold: float = 0.9

    @cached_property
    def builder(self) -> planar.WorldBuilder:
        scale = 0.1
        block_length = 4
        agent_radius = 0.5
        builder = planar.WorldBuilder(1., 1.)
        builder.add_body(planar.Body(name="agent", geom=[
            planar.Circle(
                radius=scale*agent_radius, 
                mass=1.0, color=canvas.colors.LightGreen
            )], pos=(0.5, 0.5)))
        builder.add_body(planar.Body(name="block", geom=[
            planar.Box(
                half_size=(scale*block_length/2, scale/2),
                mass=1.0, pos=(0., -scale/2),
                color=canvas.colors.LightSlateGray
            ),
            planar.Box(
                half_size=(scale/2, scale*(block_length - 1)/2),
                mass=1.0, pos=(0., -scale-scale*(block_length - 1)/2),
                color=canvas.colors.LightSlateGray
            )], pos=(-0.5, -0.5))
        )
        return builder

    def sample_action(self, rng_key: jax.Array):
        pos_agent = jax.random.randint(rng_key, (2,), 50, 450).astype(float)
        return pos_agent
    
    def reward(self, state):
        return jax.pure_callback(
            PushTEnv._callback_reward,
            jax.ShapeDtypeStruct((), jnp.float32),
            self, state
        )
    
    @partial(jax.jit, static_argnums=(2,3))
    def _render_image(self, state, width, height):
        image = 0.95*jnp.ones((width, height, 3))
        world = self.builder.renderable(state)

        translation = (self.builder.world_half_x, -self.builder.world_half_y)
        scale = (width/(2*self.builder.world_half_x),
                 -height/(2*self.builder.world_half_y))
        world = canvas.transform(world,
            translation=translation,
            scale=scale
        )
        image = canvas.paint(image, world)
        return image

    def render(self, config: RenderConfig, state: planar.PlanarState) -> jax.Array:
        state = self.builder.extract_state(state.mjx_data)
        if type(config) == ImageRender:
            return self._render_image(state, config.width, config.height)
        elif type(config) == HtmlRender:
            raise RuntimeError("Html rendering not supported!")
        
@struct.dataclass
class PushTPositionObs:
    agent_pos: jnp.array
    block_pos: jnp.array
    block_rot: jnp.array

@struct.dataclass
class PositionObsTransform(Transform):
    def transform_policy(self, policy):
        return PositionObsPolicy(policy)

@struct.dataclass
class PositionObsPolicy(Policy):
    policy: Policy

    @property
    def rollout_length(self):
        return self.policy.rollout_length
    
    def __call__(self, input):
        obs = input.observation
        obs = PushTPositionObs(
            obs.agent.position,
            obs.block.position,
            obs.block.angle
        )
        input = struct.replace(input, observation=obs)
        return self.policy(input)

# A state-feedback adapter for the PushT environment
# Will run a PID controller under the hood
@struct.dataclass
class PositionControlTransform(Transform):
    k_p : float = 100
    k_v : float = 20

    def transform_policy(self, policy):
        return PositionControlPolicy(policy, self.k_p, self.k_v)

@struct.dataclass
class PositionControlPolicy(Policy):
    policy: Policy
    k_p : float = 100
    k_v : float = 20

    @property
    def rollout_length(self):
        return self.policy.rollout_length

    def __call__(self, input):
        obs = input.observation
        output = self.policy(input)
        a = self.k_p * (output.action - obs.agent.position) + self.k_v * (-obs.agent.velocity)
        return struct.replace(
            output, action=a,
            info=AttrMap(output.info, target_pos=output.action)
        )