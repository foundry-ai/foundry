import foundry.core as F
import foundry.numpy as npx
import foundry.random

from foundry.core import tree

from foundry.env.core import (
    EnvWrapper, EnvironmentRegistry,
    RenderConfig, ImageRender, ImageActionsRender,
    ObserveConfig
)
from foundry.core.typing import Array

from foundry.env.transforms import (
    EnvTransform, ChainedTransform,
    MultiStepTransform
)

from . import assets

from foundry.core.dataclasses import dataclass, field, replace
from foundry.core.tree import static_property
from foundry.graphics import canvas

from foundry.env.mujoco.core import (
    MujocoEnvironment, SystemState, 
    SimulatorState, Action,
    quat_to_angle, render_2d
)

import shapely.geometry as sg
import foundry.numpy as jnp
import numpy as np
import mujoco

import jax.lax

import importlib.resources as resources

@dataclass
class PushTObs:
    agent_pos: Array = None
    agent_vel: Array = None

    block_pos: Array = None
    block_vel: Array = None

    block_rot: Array = None
    block_rot_vel: Array = None

@dataclass
class PushTEnv(MujocoEnvironment[SimulatorState]):
    goal_pos: Array = field(default_factory=lambda: jnp.zeros((2,), jnp.float32))
    goal_rot: Array = field(default_factory=lambda: jnp.array(-jnp.pi/4, jnp.float32))
    # use the mjx backend by default
    physics_backend: str = "mjx"

    success_threshold: float = 0.9

    agent_radius : float = 15 / 252
    block_scale : float = 30 /252
    world_scale : float = 2

    @static_property
    def xml(self):
        with (resources.files(assets) / "pusht.xml").open("r") as f:
            xml = f.read()
        com = 0.5*(self.block_scale/2) + 0.5*(self.block_scale + 1.5*self.block_scale)
        return xml.format(
            agent_radius=self.agent_radius,
            world_scale=self.world_scale,
            half_world_scale=self.world_scale/2,
            # the other constants needed for the block
            block_scale=self.block_scale,
            half_block_scale=self.block_scale/2,
            double_block_scale=2*self.block_scale,
            one_and_half_block_scale=1.5*self.block_scale,
            two_and_half_block_scale=2.5*self.block_scale,
            com_offset=com
        )
    
    @static_property
    def model(self):
        return mujoco.MjModel.from_xml_string(self.xml)

    @F.jit
    def reset(self, rng_key : Array) -> SimulatorState:
        a_pos, b_pos, b_rot, c = foundry.random.split(rng_key, 4)
        agent_pos = foundry.random.uniform(a_pos, (2,), minval=-0.8, maxval=0.8)
        block_rot = foundry.random.uniform(b_pos, (), minval=-jnp.pi, maxval=jnp.pi)
        block_pos = foundry.random.uniform(b_rot, (2,), minval=-0.4, maxval=0.4)
        # re-generate block positions while the block is too close to the agent
        min_radius = self.block_scale*2*jnp.sqrt(2) + self.agent_radius
        def gen_pos(carry):
            rng_key, _ = carry
            rng_key, sk = foundry.random.split(rng_key)
            return (rng_key, foundry.random.uniform(sk, (2,), minval=-0.4, maxval=0.4))
        _, block_pos = jax.lax.while_loop(
            lambda s: jnp.linalg.norm(s[1] - agent_pos) < min_radius,
            gen_pos, (c, block_pos)
        )
        qpos = jnp.concatenate([agent_pos, block_pos, block_rot[jnp.newaxis]])
        return self.full_state(SystemState(
            jnp.zeros((), dtype=jnp.float32), 
            qpos, 
            jnp.zeros_like(qpos), 
            jnp.zeros((0,), dtype=jnp.float32)
        ))
    
    @F.jit
    def observe(self, state, config : ObserveConfig | None = None):
        if config is None: config = PushTObs()
        data = self.simulator.system_data(state)
        if isinstance(config, PushTObs):
            return PushTObs(
                # Extract agent pos, vel
                agent_pos=data.xpos[1,:2],
                agent_vel=data.cvel[1,3:5],
                # Extract block pos, vel, angle, angular vel
                block_pos=data.xpos[2,:2],
                block_rot=quat_to_angle(data.xquat[2,:4]),
                block_vel=data.cvel[2,3:5],
                block_rot_vel=data.cvel[2,2],
            )
        elif isinstance(config, PushTAgentPos):
            return data.xpos[1,:2]
        else:
            raise ValueError("Unsupported observation type")

    # For computing the reward
    def _block_points(self, pos, rot):
        center_a, hs_a = jnp.array([0, -self.block_scale/2], dtype=jnp.float32), \
                jnp.array([2*self.block_scale, self.block_scale/2], dtype=jnp.float32)
        center_b, hs_b = jnp.array([0, -2.5*self.block_scale], dtype=jnp.float32), \
                        jnp.array([self.block_scale/2, 1.5*self.block_scale], dtype=jnp.float32)

        points = jnp.array([
            center_a + jnp.array([hs_a[0], -hs_a[1]], dtype=jnp.float32),
            center_a + hs_a,
            center_a + jnp.array([-hs_a[0], hs_a[1]], dtype=jnp.float32),
            center_a - hs_a,
            center_b + jnp.array([-hs_b[0], hs_b[1]], dtype=jnp.float32),
            center_b - hs_b,
            center_b + jnp.array([hs_b[0], -hs_b[1]], dtype=jnp.float32),
            center_b + hs_b
        ])
        rotM = jnp.array([
            [jnp.cos(rot), -jnp.sin(rot)],
            [jnp.sin(rot), jnp.cos(rot)]
        ], dtype=jnp.float32)
        points = jax.vmap(lambda v: rotM @ v)(points)
        return points + pos

    @staticmethod
    def _overlap(pointsA, pointsB):
        polyA = sg.Polygon(pointsA)
        polyB = sg.Polygon(pointsB)
        return jnp.array(polyA.intersection(polyB).area / polyA.area, dtype=jnp.float32)

    @F.jit
    def reward(self, state : SimulatorState, 
                action : Action, 
                next_state : SimulatorState):
        obs = self.observe(next_state)
        goal_points = self._block_points(self.goal_pos, self.goal_rot)
        points = self._block_points(obs.block_pos, obs.block_rot)
        overlap = jax.pure_callback(
            PushTEnv._overlap,
            jax.ShapeDtypeStruct((), jnp.float32),
            goal_points, points
        )
        return jnp.minimum(overlap, self.success_threshold) / self.success_threshold

    @F.jit
    def combined_reward(self, states: SimulatorState, actions: Action):
        prev_states = tree.map(lambda x: x[:-1], states)
        next_states = tree.map(lambda x: x[1:], states)
        if tree.axis_size(states, 0) == tree.axis_size(actions, 0):
            actions = tree.map(lambda x: x[:-1], actions)
        rewards = F.vmap(self.reward)(prev_states, actions, next_states)
        # use the max reward over the trajectory
        return npx.max(rewards)

    @F.jit
    def render(self, state : SimulatorState, config : RenderConfig | None = None): 
        if config is None: config = ImageRender(width=256, height=256)
        if isinstance(config, ImageRender):
            data = self.simulator.system_data(state)
            image = jnp.ones((config.height, config.width, 3))
            target = render_2d(
                self.model, data, 
                config.width, config.height,
                2, 2,
                body_custom={2: (self.goal_pos, self.goal_rot, jnp.array([0, 1, 0]))}
            )
            world = render_2d(
                self.model, data, 
                config.width, config.height,
                2, 2
            )
            image = canvas.paint(image, target, world)
            if isinstance(config, ImageActionsRender) and config.actions is not None:
                def draw_action_chunk(action_chunk):
                    T = action_chunk.shape[0]
                    colors = jnp.array((jnp.arange(T)/T, jnp.zeros(T), jnp.ones(T))).T
                    circles = canvas.fill(
                        canvas.circle(action_chunk, 0.02*jnp.ones(T)),
                        color=colors
                    )
                    circles = canvas.stack_batch(circles)
                    circles = canvas.transform(circles,
                        translation=(1,-1),
                        scale=(config.width/2, -config.height/2)
                    )
                    return circles
                circles = draw_action_chunk(config.actions)
                image = canvas.paint(image, circles)
            return image
        else:
            raise ValueError("Unsupported render config")

@dataclass
class PushTAgentPos:
    pass

@dataclass
class PushTPosObs:
    agent_pos: jnp.array = None
    block_pos: jnp.array = None
    block_rot: jnp.array = None

@dataclass
class PushTKeypointObs:
    agent_pos: jnp.array = None
    block_pos: jnp.array = None
    block_end: jnp.array = None

@dataclass
class PushTKeypointRelObs:
    agent_block_pos: jnp.array = None
    agent_block_end: jnp.array = None
    rel_block_pos: jnp.array = None
    rel_block_end: jnp.array = None
        
# A state-feedback adapter for the PushT environment
# Will run a PID controller under the hood
@dataclass
class PositionalControlTransform(EnvTransform):
    k_p : float = 15
    k_v : float = 2
    
    def apply(self, env):
        return PositionalControlEnv(env, self.k_p, self.k_v)

@dataclass
class PositionalObsTransform(EnvTransform):
    def apply(self, env):
        return PositionalObsEnv(env)

@dataclass
class KeypointObsTransform(EnvTransform):
    def apply(self, env):
        return KeypointObsEnv(env)

@dataclass
class RelKeypointObsTransform(EnvTransform):
    def apply(self, env):
        return RelKeypointObsEnv(env)

@dataclass
class PositionalControlEnv(EnvWrapper):
    k_p : float = 50
    k_v : float = 2

    def step(self, state, action, rng_key=None):
        obs = self.base.observe(state, PushTObs())
        if action is not None:
            a = self.k_p * (action - obs.agent_pos) + self.k_v * (-obs.agent_vel)
        else: 
            a = jnp.zeros((2,), dtype=jnp.float32)
        return self.base.step(state, a, None)


@dataclass
class PositionalObsEnv(EnvWrapper):
    def observe(self, state, config=None):
        if config is None: config = PushTPosObs
        if config != PushTPosObs:
            return self.base.observe(state, config)
        obs = self.base.observe(state, PushTObs())
        return PushTPosObs(
            agent_pos=obs.agent_pos,
            block_pos=obs.block_pos,
            block_rot=obs.block_rot
        )

@dataclass
class KeypointObsEnv(EnvWrapper):
    def observe(self, state, config=None):
        if config is None: config = PushTKeypointObs()
        if not isinstance(config, PushTKeypointObs):
            return self.base.observe(state, config)
        obs = self.base.observe(state, PushTObs())
        rotM = jnp.array([
            [jnp.cos(obs.block_rot), -jnp.sin(obs.block_rot)],
            [jnp.sin(obs.block_rot), jnp.cos(obs.block_rot)]
        ], dtype=obs.block_pos.dtype)
        end = rotM @ jnp.array([0, -4*self.block_scale], dtype=rotM.dtype) + obs.block_pos
        return PushTKeypointObs(
            agent_pos=obs.agent_pos,
            block_pos=obs.block_pos,
            block_end=end
        )

@dataclass
class RelKeypointObsEnv(EnvWrapper):
    def observe(self, state, config=None):
        if config is None: config = PushTKeypointRelObs()
        if not isinstance(config, PushTKeypointRelObs):
            return self.base.observe(state, config)
        obs = self.base.observe(state, PushTObs())
        rotM = jnp.array([
            [jnp.cos(obs.block_rot), -jnp.sin(obs.block_rot)],
            [jnp.sin(obs.block_rot), jnp.cos(obs.block_rot)]
        ])
        end = rotM @ jnp.array([0, -4*self.block_scale], dtype=rotM.dtype) + obs.block_pos
        rotM = jnp.array([
            [jnp.cos(self.goal_rot), -jnp.sin(self.goal_rot)],
            [jnp.sin(self.goal_rot), jnp.cos(self.goal_rot)]
        ])
        goal_end = rotM @ jnp.array([0, -4*self.block_scale], dtype=rotM.dtype) + self.goal_pos
        return PushTKeypointRelObs(
            agent_block_pos=obs.agent_pos - obs.block_pos,
            agent_block_end=obs.agent_pos - end,
            rel_block_pos=obs.block_pos - self.goal_pos,
            rel_block_end=end - goal_end,
        )

def register_all(registry, prefix=None):
    registry.register("pusht", PushTEnv, prefix=prefix)
