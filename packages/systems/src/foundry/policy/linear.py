from foundry.core import Array
from foundry.core.dataclasses import dataclass
from foundry.policy import Policy, PolicyInput, PolicyOutput

class LinearPolicy(Policy):
    K: Array

    def __call__(input: PolicyInput) -> PolicyOutput:
        obs = input.observation
        return PolicyOutput(K @ obs)