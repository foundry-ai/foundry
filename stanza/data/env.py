from .rng import RNGDataset
from typing import Any
from stanza.dataset import Dataset, INFINITE, UNKNOWN

from stanza.envs import Environment

from stanza.policies import RandomPolicy, Policy
from stanza.util.dataclasses import dataclass, field, replace

from jax.random import PRNGKey

import stanza.policies
import jax

@dataclass(jax=True)
class TrajectoryEntry:
    trajectory_id: int
    timestep: int
    env_state : Any
    policy_output : PolicyOutput

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
    env_state : Any
    policy_output : PolicyOutput

@dataclass(jax=True)
class TrajGenerator(Dataset):
    rng_key: PRNGKey
    trajectory_id: int

    env: Environment
    policy: Policy
    traj_length: int
    is_finished: Callable = None

    @property
    def start(self):
        rng_key, r_sk, p_sk = jax.random.split(rng_key, 3)
        state = env.reset(r_sk)
        output = self.policy(PolicyInput(state, None, p_sk))
        return RolloutIterator(
            rng_key, 0, state, output
        )
    
    def remaining(self, iterator)
        return jax.lax.cond(
            iterator.timestep + 1 >= self.traj_length,
            lambda: 0,
            lambda: UNKNOWN
        )

    def next(self, iterator):
        pass

@dataclass(jax=True)
class EnvDataset(RNGDataset):
    env: Environment
    policy: Policy
    traj_length: int
    is_finished: Callable = None

    def get(self, iterator):
        return EnvTrajGenerator(iterator,
            self.env, self.policy, self.traj_length,
            self.is_finished)

@dataclass(jax=True)
class TrajectoryDataset(Dataset):
    entries: Any
    start_indices: jnp.array