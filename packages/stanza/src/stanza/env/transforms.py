import abc
import jax

from stanza.dataclasses import dataclass, field
from stanza.env import EnvWrapper

class EnvTransform(abc.ABC):
    @abc.abstractmethod
    def apply(self, env): ...

@dataclass
class ChainedTransform(EnvTransform):
    transforms: list[EnvTransform]

    def apply(self, env):
        for t in self.transforms:
            env = t.apply(env)
        return env

@dataclass
class MultiStepTransform(EnvTransform):
    steps: int = field(default=1, pytree_node=False)

    def apply(self, env):
        return MultiStepEnv(env, self.steps)

@dataclass
class MultiStepEnv(EnvWrapper):
    steps: int = field(default=1, pytree_node=False)

    @jax.jit
    def step(self, state, action, rng_key=None):
        keys = jax.random.split(rng_key, self.steps) \
            if rng_key is not None else None
        def step_fn(state, key):
            state = self.base.step(state, action, key)
            return state, None
        state, _ = jax.lax.scan(step_fn, state, keys, length=self.steps)
        return state