import typing
import jax
import abc

from stanza.dataclasses import dataclass, field
from stanza.util.registry import Registry, from_module
from typing import Optional, Sequence

State = typing.TypeVar("State")
ReducedState = typing.TypeVar("ReducedState")
Action = typing.TypeVar("Action")
Observation = typing.TypeVar("Observation")
Render = typing.TypeVar("Render")

class RenderConfig(typing.Generic[Render]): ...
class ObserveConfig(typing.Generic[Observation]): ...

class Environment(typing.Generic[State, ReducedState, Action]):
    # By default ReducedState = FullState
    # The notion of "full" and "reduced states" are useful for
    # environments that have cached information in the state.
    # (active contacts, contact forces, etc.)
    # As a result we may only approximately have:
    # step(full_state(reduced_state(s))) ~= step(s)
    # up to some solver error tolerance.

    def full_state(self, reduced_state: ReducedState) -> State:
        return reduced_state
    def reduce_state(self, full_state: State) -> ReducedState:
        return full_state

    def sample_state(self, rng_key : jax.Array) -> State:
        raise NotImplementedError()

    def sample_action(self, rng_key : jax.Array) -> Action:
        raise NotImplementedError()

    def reset(self, rng_key : jax.Array) -> State:
        raise NotImplementedError()

    def step(self, state : State, action : Action,
             rng_key : Optional[jax.Array] = None) -> State:
        raise NotImplementedError()
            
    def observe(self, state: State,
            config: ObserveConfig[Observation] | None = None) -> Observation:
        raise NotImplementedError()

    def reward(self, state: State,
               action : Action, next_state : State) -> jax.Array:
        raise NotImplementedError()

    def cost(self, states: State, actions: Action) -> jax.Array:
        raise NotImplementedError()

    def is_finished(self, state: State) -> jax.Array:
        raise NotImplementedError()

    def render(self, state : State,
               config: RenderConfig[Render] | None = None) -> Render:
        raise NotImplementedError()

@dataclass
class EnvWrapper(Environment[State, ReducedState, Action]):
    base: Environment[State, ReducedState, Action]

    # We need to manually override these since
    # they are already defined in Environment
    # and therefore not subject to __getattr__

    def full_state(self, reduced_state: ReducedState) -> State:
        return self.base.full_state(reduced_state)
    def reduce_state(self, full_state: State) -> ReducedState:
        return self.base.reduce_state(full_state)

    def sample_state(self, rng_key : jax.Array) -> State:
        return self.base.sample_state(rng_key)

    def sample_action(self, rng_key : jax.Array) -> Action:
        return self.base.sample_action(rng_key)

    def reset(self, rng_key : jax.Array) -> State:
        return self.base.reset(rng_key)

    def step(self, state : State, action : Action,
             rng_key : Optional[jax.Array] = None) -> State:
        return self.base.step(state, action, rng_key)

    def observe(self, state: State,
            config: ObserveConfig[Observation] | None = None) -> Observation:
        return self.base.observe(state, config)

    def reward(self, state: State,
               action : Action, next_state : State) -> jax.Array:
        return self.base.reward(state, action, next_state)

    def cost(self, states: State, actions: Action) -> jax.Array:
        return self.base.cost(states, actions)

    def is_finished(self, state: State) -> jax.Array:
        return self.base.is_finished(state)

    def render(self, state : State,
               config: RenderConfig[Render] | None = None) -> Render:
        return self.base.render(state, config)

    def __getattr__(self, name):
        print(name)
        return getattr(self.base, name)

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