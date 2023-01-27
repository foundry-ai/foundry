from .rng import RNGDataset

from ode.policy import SampleRandom

import ode.envs
import jax

class EnvDataset(RNGDataset):
    def __init__(self, rng_key, env, traj_length, policy=None, jacobians=False):
        rng_key, sk = jax.random.split(rng_key)
        super().__init__(rng_key)
        self.env = env
        # If policy is not specified, sample random
        # actions from the environment
        if policy is None:
            policy = SampleRandom(sk, env.sample_action)
        self.policy = policy
        self.traj_length = traj_length
        self.jacobians = jacobians

    def get(self, iterator):
        rng = super().get(iterator)

        traj = ode.envs.rollout_policy(self.env.step,
            self.env.reset(rng),
            self.traj_length, self.policy, jacobians=self.jacobians)
        return traj