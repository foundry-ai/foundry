from .rng import RNGDataset
from stanza.dataset import PyTreeDataset

from stanza.envs import Environment

from stanza.policies import RandomPolicy, Policy
from stanza.util.dataclasses import dataclass, field
import stanza.policies
import jax

def _to_state_action(sample):
    states = jax.tree_util.tree_map(lambda x: x[:-1], sample.states)
    actions = sample.actions
    return states, actions

@dataclass(jax=True)
class EnvDataset(RNGDataset):
    env: Environment = field(jax_static=True)
    traj_length: int = field(jax_static=True)
    policy: Policy

    def as_state_actions(self):
        m = self.map(_to_state_action)
        return m.flatten()

    def get(self, iterator):
        rng = super().get(iterator)
        traj = stanza.policies.rollout(self.env.step,
            self.env.reset(rng),
            length=self.traj_length, policy=self.policy)
        return traj