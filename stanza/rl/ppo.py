from stanza.dataclasses import dataclass

@dataclass(jax=True)
class PPO:
    gamma: float = 0.9
    def update(self, state):
        pass