from .rng import RNGDataset
from stanza.dataset import PyTreeDataset

from stanza.envs import Environment

from stanza.policies import RandomPolicy, Policy
from stanza.util.dataclasses import dataclass, field, replace
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

    @staticmethod
    def as_state_actions(dataset):
        m = dataset.map(_to_state_action)
        return m.flatten()

    def get(self, iterator):
        rng = super().get(iterator)
        rng, sk = jax.random.split(rng)

        # TODO: There should be a cleaner
        # way to have a policy with an rng seed key
        if hasattr(self.policy, 'rng_key'):
            policy = replace(self.policy, rng_key=sk)
        else:
            policy = self.policy
        traj = stanza.policies.rollout(self.env.step,
            self.env.reset(rng),
            length=self.traj_length, policy=policy)
        return traj