from stanza.envs import Environment
from stanza.policies import Policy, PolicyOutput, PolicyTransform, \
                            chain_transforms, \
                            SampleRateTransform, ChunkTransform
from stanza.util.dataclasses import dataclass, field, replace
from stanza.util.attrdict import AttrMap
from stanza.data.trajectory import (
    Timestep, IndexedTrajectoryData, TrajectoryIndices
)
from stanza.envs.pymunk import PyMunkWrapper, System, Bodies, BodyDef, BodyState

from stanza.data import Data
from functools import partial
from jax.random import PRNGKey

import pymunk
import shapely.geometry as sg
import jax.numpy as jnp
import jax.random

@dataclass(jax=True)
class PushTEnv(PyMunkWrapper):
    width: float = 100.0
    height: float = 100.0
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

    def step(self, state, action, rng_key):
        return jax.pure_callback(PushTEnv._callback_step, state, self, state, action)

    def score(self, state):
        return jax.pure_callback(
            PushTEnv._callback_score,
            jax.ShapeDtypeStruct((), jnp.float32),
            self, state
        )

    def _system_def(self, rng_key):
        z = jnp.zeros(())
        z2 = jnp.zeros((2,))
        pos_key, block_key, rot_key = jax.random.split(rng_key, 3)
        pos_agent = jax.random.randint(pos_key, (2,), 50, 450).astype(float)
        agent_state = BodyState(pos_agent, z2, z, z)
        pos_block = jax.random.randint(block_key, (2,), 100, 400).astype(float)
        rot_block = jax.random.uniform(rot_key, minval=-jnp.pi, maxval=jnp.pi)
        block_state = BodyState(pos_block, z2, rot_block, z)

        agent_shape = pymunk.Circle(radius=15)
        agent_shape.color = (65/255, 105/255, 255/255)

        # Add the circle agent
        agent = self._add_circle(space, agent_state, 15,
                color=(65/255, 105/255, 225/255))
        block = self._add_tee(space, block_state,
                color=(119/255, 136/255, 153/255))
        # Add collision handeling
        # _ = space.add_collision_handler(0, 0)
        return space, agent, block

    def _add_tee(self, space, state,
                 scale=30, color=None, mask=pymunk.ShapeFilter.ALL_MASKS(), z=None):
        mass = 1
        length = 4
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
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.z = z
        shape2.z = z
        shape1.color = color
        shape2.color = color
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.friction = 1
        # self._set_body_state(body, state)
        space.add(body, shape1, shape2)

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