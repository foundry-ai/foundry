from stanza.dataclasses import dataclass, field

@dataclass(jax=True)
class RL:
    total_iterations: int = field(jax_static=True)

    def init_state(self, policy):
        pass

    def update(self, state):
        pass

    def run(self, state):
        pass