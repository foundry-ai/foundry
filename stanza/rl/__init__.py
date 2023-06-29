from stanza.dataclasses import dataclass, field
from stanza.util.attrdict import AttrMap
from stanza.policies import PolicyInput, PolicyOutput

from typing import Callable

@dataclass(jax=True)
class ACPolicy:
    actor_critic: Callable

    def __call__(self, input: PolicyInput) -> PolicyOutput:
        pi, value = self.actor_critic(input.observation)
        action = pi.sample(input.rng_key)
        log_prob = pi.log_prob(action)
        return PolicyOutput(
            action, log_prob, 
            AttrMap(log_prob=log_prob, value=value)
        )