from .rng import RNGDataset

from stanza.policies import RandomPolicy
import stanza.policies
import jax

class EnvDataset(RNGDataset):
    def __init__(self, rng_key, env, traj_length, policy=None, last_state=True):
        rng_key, sk = jax.random.split(rng_key)
        super().__init__(rng_key)
        self.env = env
        # If policy is not specified, sample random
        # actions from the environment
        if policy is None:
            policy = RandomPolicy(sk, env.sample_action)
        self.policy = policy
        self.last_state = last_state
        self.traj_length = traj_length

    def get(self, iterator):
        rng = super().get(iterator)
        traj = stanza.policies.rollout(self.env.step,
            self.env.reset(rng),
            length=self.traj_length, policy=self.policy, last_state=self.last_state)
        return traj