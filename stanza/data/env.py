from stanza.util.random import PRNGDataset
from typing import Any, Callable
from stanza.data import Data, UNKNOWN

from stanza.envs import Environment
from stanza.policies import Policy, PolicyOutput, PolicyInput
from stanza.util.dataclasses import dataclass, field, replace

from jax.random import PRNGKey
import jax
import jax.numpy as jnp

@dataclass(jax=True)
class Timestep:
    observation: Any
    action: Any

# A dataset of Trajectories
# has a get() method which
# returns a Dataset per-trajectory
# which can themselves be iterated
# to get each of the state entries
# Calling flatten() on a trajectory

@dataclass(jax=True)
class RolloutIterator:
    rng_key: PRNGKey
    timestep: int
    end: bool
    env_state : Any
    policy_output : PolicyOutput

@dataclass(jax=True)
class TrajGenerator(Data):
    rng_key: PRNGKey
    trajectory_id: int

    env: Environment
    policy: Policy
    traj_length: int
    is_finished: Callable = None

    @property
    def start(self):
        rng_key, r_sk, p_sk = jax.random.split(rng_key, 3)
        state = self.env.reset(r_sk)
        output = self.policy(PolicyInput(state, None, p_sk))
        return RolloutIterator(
            rng_key, 0, False, state, output
        )

    def is_end(self, iterator):
        return jnp.logical_or(
            iterator.timestep >= self.traj_length,
            iterator.end
        )

    def remaining(self, iterator):
        return UNKNOWN

    def next(self, iterator):
        pass

@dataclass(jax=True)
class EnvGenerator(PRNGDataset):
    env: Environment
    policy: Policy
    traj_length: int
    is_finished: Callable = None

    def get(self, iterator):
        return TrajGenerator(iterator,
            self.env, self.policy, self.traj_length,
            self.is_finished)

@dataclass(jax=True)
class TrajectoryIndices:
    start_index: int
    end_index: int

@dataclass(jax=True)
class TrajectoryData(Data):
    trajectory_indices: Data
    timesteps: Data

    @property
    def start(self):
        return self.trajectory_indices.start

    def remaining(self, iterator):
        return self.trajectory_indices.remaining(iterator)

    def is_end(self, iterator):
        return self.trajectory_indices.is_end(iterator)
    
    def get(self, iterator):
        idx = self.trajectory_indices.get(iterator)
    
    # Override slice()
    # to slice just trajectory_info
    # We can't slice timesteps due to step,
    # so for now just keep all of the timesteps around
    def slice(self, start, stop, step):
        return TrajectoryData(
            self.trajectory_indices.slice(
                start, stop, step
            ),
            self.timesteps
        )
    
    def flatten(self):
        return self.timesteps
    
    # TODO: For now we just hack
    # chunk() into here...
    # we need to figure out a way to make
    # this play cleanly though
    def chunk(input_chunk_size=None, output_chunk_size=None):
        return NOne

@dataclass(jax=True)
class ChunkedTrajectory(Data):
    pass

# TODO: Old chunked trajectories
# @dataclass(jax=True)
# class ChunkedTrajectory(Data):
#     trajectory: Data
#     input_chunk_size: int = field(default=None, jax_static=True)
#     output_chunk_size: int = field(default=None, jax_static=True)

#     def start(self):
#         i = self.input_chunk_size \
#             if self.input_chunk_size is not None else 1
#         o = self.output_chunk_size \
#             if self.output_chunk_size is not None else 1
#         start = self.trajectory.start
#         def scan_fn(i, _):
#             n = self.trajectory.next(i)
#             return n, n
#         _, post = jax.lax.scan(scan_fn, start, None, length=o-1)
#         # stack start i times, followed by *post*
#         return jax.tree_util.tree_map(
#             lambda s, p: jnp.concatenate(
#                 (jnp.repeat(s[jnp.newaxis,...], i, 0), p)
#             ), start, post
#         )

#     def remaining(self, iterator):
#         i = self.input_chunk_size \
#             if self.input_chunk_size is not None else 1
#         mid_iterator = jax.tree_util.tree_map(
#             lambda s: s[i], iterator)
#         return self.trajectory.remaining(mid_iterator)

#     def is_end(self, iterator):
#         i = self.input_chunk_size \
#             if self.input_chunk_size is not None else 1
#         mid_iterator = jax.tree_util.tree_map(
#             lambda s: s[i], iterator)
#         return self.trajectory.is_end(mid_iterator)
    
#     def next(self, iterator):
#         pass

#     def get(self, iterator):
#         i = self.input_chunk_size \
#             if self.input_chunk_size is not None else 1
#         input_iterators = jax.tree_util.tree_map(
#             lambda x: x[:i], iterator
#         )
#         output_iterators = jax.tree_util.tree_map(
#             lambda x: x[i-1:], iterator
#         )
#         vget = jax.vamp(self.trajectory.get)
#         input = vget(input_iterators).observation
#         output = vget(output_iterators).action
#         return Timestep(input, output)

# class ChunkedTrajectories(Data):
#     pass