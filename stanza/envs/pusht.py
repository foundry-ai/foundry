from stanza.envs import Environment
import stanza.policies as policies
from stanza.policies import Policy, PolicyOutput, PolicyTransform
from stanza.util.dataclasses import dataclass, field, replace
from stanza.util.attrdict import AttrMap
from stanza.data.trajectory import (
    Timestep, IndexedTrajectoryData, TrajectoryIndices
)
from stanza.envs.pymunk import PyMunkWrapper, BodyState

from stanza.data import Data
from functools import partial
from jax.random import PRNGKey

import pymunk
import numpy as np
import shapely.geometry as sg
import jax.numpy as jnp
import jax.random

@dataclass(jax=True)
class PushTEnv(PyMunkWrapper):
    width: float = 512.0
    height: float = 512.0
    sim_hz: float = 100.0

    goal_pose : BodyState = field(
        default_factory=lambda: BodyState(
            jnp.array([256.,256.]),
            jnp.array([0.,0.]),
            jnp.array(jnp.pi/4),
            jnp.array(0)
        ))
    success_threshold: float = 0.9

    def sample_action(self, rng_key):
        pos_agent = jax.random.randint(rng_key, (2,), 50, 450).astype(float)
        return pos_agent

    def sample_state(self, rng_key):
        return self.reset(rng_key)

    def reward(self, state):
        return jax.pure_callback(
            PushTEnv._callback_reward,
            jax.ShapeDtypeStruct((), jnp.float32),
            self, state
        )
    
    def _callback_reward(self, state):
        space, _, block = self._setup_space(state.agent, state.block)
        goal = self._add_tee(space, self.goal_pose, color=(0,1,0), z=-1)

        goal_geom = pymunk_to_shapely(goal)
        block_geom = pymunk_to_shapely(block)
        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        reward = jnp.clip(coverage / self.success_threshold, 0., 1.)
        return reward

    def _build_space(self, rng_key):
        pos_key, block_key, rot_key = jax.random.split(rng_key, 3) \
            if rng_key is not None else (None, None, None)

        space = pymunk.Space()
        # add the walls
        walls = [
            pymunk.Segment(space.static_body, (5, 506), (5, 5), 2),
            pymunk.Segment(space.static_body, (5, 5), (506, 5), 2),
            pymunk.Segment(space.static_body, (506, 5), (506, 506), 2),
            pymunk.Segment(space.static_body, (5, 506), (506, 506), 2),
        ]
        space.add(*walls)

        # make the agent
        agent = pymunk.Body()
        agent.friction = 1
        agent.name = 'agent'
        agent_shape = pymunk.Circle(agent, radius=15)
        agent_shape.color = (65/255, 105/255, 255/255)
        pos_agent = jax.random.randint(pos_key,
                        (2,), 50, 450).astype(float) \
                            if pos_key is not None else np.array([0.,0.])
        agent.position = (pos_agent[0].item(), pos_agent[1].item())
        space.add(agent, agent_shape)

        # Add the block
        pos_block = jax.random.randint(block_key, (2,), 200, 400).astype(float) \
                            if block_key is not None else np.array([0.,0.])
        rot_block = jax.random.uniform(rot_key, minval=-jnp.pi, maxval=jnp.pi) \
                            if rot_key is not None else np.array(0.)
        block_color = (119/255, 136/255, 153/255)
        self._add_block(space, pos_block, rot_block, block_color)
        return space
    
    def _add_block(self, space, pos_block, rot_block, block_color,
                mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass, length, scale = 1, 4, 30
        vertices1 = [(-length*scale/2, scale),
                                 ( length*scale/2, scale),
                                 ( length*scale/2, 0),
                                 (-length*scale/2, 0)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [(-scale/2, scale),
                     (-scale/2, length*scale),
                     ( scale/2, length*scale),
                     ( scale/2, scale)]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        body.name = 'block'
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = block_color
        shape2.color = block_color
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.friction = 1

        body.position = (pos_block[0].item(), pos_block[1].item())
        body.angle = rot_block.item()

        space.add(body, shape1, shape2)
    
    def _space_action(self, space, action, rng_key):
        agent = space.bodies[0]
        dt = 1.0 / self.sim_hz
        agent.velocity += action * dt
        return space
    
    def teleop_policy(self, interface):
        def policy(input):
            size = jnp.array([interface.width, interface.height])
            p = interface.mouse_position / size
            return PolicyOutput(
                action=p
            )
        return policies.chain_transforms(
            PositionControlTransform(),
        )(policy)

def builder(name):
    return PushTEnv()


@dataclass(jax=True)
class PushTPositionObs:
    agent_pos: jnp.array
    block_pos: jnp.array
    block_rot: jnp.array

@dataclass(jax=True)
class PositionObsTransform(PolicyTransform):
    def transform_policy(self, policy):
        return PositionObsPolicy(policy)

@dataclass(jax=True)
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
        input = replace(input, observation=obs)
        return self.policy(input)

# A state-feedback adapter for the PushT environment
# Will run a PID controller under the hood
@dataclass(jax=True)
class PositionControlTransform(PolicyTransform):
    k_p : float = 100
    k_v : float = 20

    def transform_policy(self, policy):
        return PositionControlPolicy(policy, self.k_p, self.k_v)

@dataclass(jax=True)
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
        return replace(
            output, action=a,
            info=AttrMap(output.info, target_pos=output.action)
        )

# ----- The expert dataset ----

def expert_data():
    import gdown
    import os
    cache = os.path.join(os.getcwd(), '.cache')
    os.makedirs(cache, exist_ok=True)
    dataset_path = os.path.join(cache, 'pusht_data.zarr.zip')
    if not os.path.exists(dataset_path):
        id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
        gdown.download(id=id, output=dataset_path, quiet=False)
    import zarr
    with zarr.open(dataset_path, "r") as data:
        # Read in all of the data
        state = jnp.array(data['data/state'])
        actions = jnp.array(data['data/action'])
        episode_ends = jnp.array(data['meta/episode_ends'])
        episode_starts = jnp.roll(episode_ends, 1)
        episode_starts = episode_starts.at[0].set(0)
    # fill in zeros for the missing state data
    z = jnp.zeros((state.shape[0],))
    z2 = jnp.zeros((state.shape[0],2))
    agent_pos = state[:,:2]
    block_pos = state[:,2:4]
    block_rot = state[:,4]
    states = PushTPositionObs(
        agent_pos,
        block_pos,
        block_rot
    )
    timesteps = Timestep(
        states,
        actions
    )
    indices = TrajectoryIndices(
        episode_starts,
        episode_ends
    )
    return IndexedTrajectoryData(
        Data.from_pytree(indices),
        Data.from_pytree(timesteps)
    )

def pymunk_to_shapely(body):
    geoms = list()
    for shape in body.shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom