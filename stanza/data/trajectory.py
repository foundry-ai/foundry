from stanza.data import Data

from stanza.dataclasses import dataclass, field
from stanza.util.attrdict import AttrMap
from typing import Any, List

@dataclass(jax=True)
class Timestep:
    observation: Any
    action: Any
    # The full state (may also be the observation)
    state: Any = None
    # Any additional per-timestep info
    info: AttrMap = field(default_factory=AttrMap)

# A dataset of Trajectories
# has a get() method which
# returns a Dataset per-trajectory
# which can themselves be iterated
# to get each of the state entries
@dataclass(jax=True)
class TrajectoryIndices:
    start_index: int
    end_index: int

@dataclass(jax=True)
class IndexedTrajectoryData(Data):
    trajectory_indices: Data
    timesteps: Data

    @property
    def start(self):
        return self.trajectory_indices.start

    def remaining(self, iterator):
        return self.trajectory_indices.remaining(iterator)

    def is_end(self, iterator):
        return self.trajectory_indices.is_end(iterator)
    
    def next(self, iterator):
        return self.trajectory_indices.next(iterator)

    def advance(self, iterator, n):
        return self.trajectory_indices.advance(iterator, n)
    
    def get(self, iterator):
        idx = self.trajectory_indices.get(iterator)
        iter = self.timesteps.advance(
            self.timesteps.start, idx.start_index)
        return self.timesteps.slice(
            iter, idx.end_index - idx.start_index)

    # Override slice()
    # to slice just trajectory_info
    # We can't slice timesteps due to step,
    # so for now just keep all of the timesteps around
    def slice(self, start_iter, len):
        return IndexedTrajectoryData(
            self.trajectory_indices.slice(
                start_iter, len
            ),
            self.timesteps
        )
    
    def shuffle(self):
        return IndexedTrajectoryData(
            self.trajectory_indices.shuffle(),
            self.timesteps
        )