import typing
import jax

from stanza.util.registry import Registry, register_module
from typing import Optional

State = typing.TypeVar("State")
Action = typing.TypeVar("Action")
Observation = typing.TypeVar("Observation")
Render = typing.TypeVar("Render")

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

class Renderer(typing.Protocol[State, Action, Render]):
    def render(self, state : State, **kwargs) -> Render: ...

EnvironmentRegistry = Registry

env_registry = EnvironmentRegistry[Environment]()
# env_registry.defer(register_module(".pusht", "env_registry"))
env_registry.defer(register_module(".linear", "env_registry"))
env_registry.defer(register_module(".quadrotor_2d", "env_registry"))