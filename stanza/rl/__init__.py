from stanza.dataclasses import dataclass, field
from stanza.util.attrdict import AttrMap
from stanza.policies import PolicyInput, PolicyOutput

from typing import Callable

@dataclass(jax=True)
class ACPolicy:
    actor_critic: Callable
    observation_normalizer: Callable = None
    action_normalizer: Callable = None

    def __call__(self, input: PolicyInput) -> PolicyOutput:
        observation = input.observation
        if self.observation_normalizer is not None:
            observation = self.observation_normalizer.normalize(observation)
        pi, value = self.actor_critic(observation)
        action = pi.sample(input.rng_key)
        log_prob = pi.log_prob(action)
        return PolicyOutput(
            action, log_prob, 
            AttrMap(log_prob=log_prob, value=value)
        )