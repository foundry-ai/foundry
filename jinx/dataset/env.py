from .rng import RNGDataset

import jinx.envs

class EnvDataset(RNGDataset):
    def __init__(self, rng_key, env, policy, traj_length):
        super().__init__(rng_key)
        self.env = env
        self.policy = policy
        self.traj_length = traj_length

    def get(self, iterator):
        rng = super().get(iterator)

        states, us = jinx.envs.rollout_policy(self.env.step,
            self.env.reset(rng),
            self.traj_length, self.policy)
        return (states, us)