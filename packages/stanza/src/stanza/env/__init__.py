import typing
import jax

from stanza.dataclasses import dataclass, field
from stanza.util.registry import Registry, from_module
from typing import Optional, Sequence

State = typing.TypeVar("State")
Action = typing.TypeVar("Action")
Observation = typing.TypeVar("Observation")
Render = typing.TypeVar("Render")

class RenderConfig(typing.Generic[Render]): ...

# Generic environment. Note that all
# environments are also adapters
@typing.runtime_checkable
class Environment(typing.Protocol[State, Action, Observation]):
    def sample_state(self, rng_key : jax.Array) -> State: ...
    def sample_action(self, rng_key : jax.Array) -> Action: ...

    def reset(self, rng_key : jax.Array) -> State: ...
    def step(self, state : State, action : Action,
             rng_key : Optional[jax.Array] = None) -> State: ...

    def observe(self, state: State) -> Observation: ...
    def reward(self, state: State,
               action : Action, next_state : State) -> jax.Array: ...
    def cost(self, states: State, actions: Action) -> jax.Array: ...
    def is_finished(self, state: State) -> jax.Array: ...

    def render(self, config: RenderConfig[Render], 
               state : State, **kwargs) -> Render: ...

@dataclass
class EnvWrapper(Environment[State, Action, Observation]):
    base: Environment[State, Action, Observation]

    def sample_state(self, rng_key : jax.Array) -> State:
        return self.base.sample_state(rng_key)
    def sample_action(self, rng_key : jax.Array) -> Action:
        return self.base.sample_action(rng_key)

    def reset(self, rng_key : jax.Array) -> State:
        return self.base.reset(rng_key)
    def step(self, state : State, action : Action,
             rng_key : Optional[jax.Array] = None) -> State:
        return self.base.step(state, action, rng_key)

    def observe(self, state: State) -> Observation:
        return self.base.observe(state)

    def reward(self, state: State,
               action : Action, next_state : State) -> jax.Array:
        return self.base.reward(state, action, next_state)
    def cost(self, states: State, actions: Action) -> jax.Array:
        return self.base.cost(states, actions)

    def is_finished(self, state: State) -> jax.Array:
        return self.base.is_finished(state)
    
    def render(self, config: RenderConfig[Render], 
               state : State, **kwargs) -> Render:
        return self.base.render(config, state, **kwargs)

@dataclass
class ImageRender(RenderConfig[jax.Array]):
    width: int = field(pytree_node=False, default=256)
    height: int = field(pytree_node=False, default=256)

@dataclass
class SequenceRender(ImageRender): ...

class HtmlRender(RenderConfig[str]): ...

EnvironmentRegistry = Registry

environments = EnvironmentRegistry[Environment]()
# env_registry.defer(register_module(".pusht", "env_registry"))
environments.extend("controls", from_module(".controls", "environments"))
environments.extend("mujoco", from_module(".mujoco", "environments"))

def create(path: str, /, **kwargs):
    return environments.create(path, **kwargs)

__all__ = [
    "State", "Action", "Observation", "Render",
    "Environment", "Renderer",
    "EnvironmentRegistry",
    "env_registry"
]

