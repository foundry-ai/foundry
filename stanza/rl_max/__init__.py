import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from typing import Callable, List, Any
import stanza
from stanza.dataclasses import dataclass, field, replace
from stanza.util.attrdict import AttrMap


@dataclass(jax=True)
class QFunctionInput:
    observation: Any
    action: Any
    q_state: Any = None
    rng_key: PRNGKey = None

@dataclass(jax=True)
class QFunctionOutput:
    q_value: Any
    q_state: Any = None
    info: AttrMap = field(default_factory=AttrMap)

@dataclass(jax=True)
class VFunctionInput:
    observation: Any
    v_state: Any = None
    rng_key: PRNGKey = None

@dataclass(jax=True)
class VFunctionOutput:
    v_value: Any
    v_state: Any = None
    info: AttrMap = field(default_factory=AttrMap)

@dataclass(jax=True)
class goalObs:
    just_observation: Any
    goal: Any

class ValueFunction:
    def __call__(self, input):
        raise NotImplementedError("Must implement __call__()")

class Qfunction:
    def __call__(self, input):
        raise NotImplementedError("Must implement __call__()")
    
# for simplicity, let us assume STATELESS 
# Q and ValueFunctions
class ValueTransform: 
    def transform_value_function(self, value_function):
        return value_function
    def __call__(self, value_function, value_state = None):
        return self.transform_value_function(value_function)

class QTransform: 
    def transform_q_function(self, q_function):
        return q_function
    
    def __call__(self, q_function, value_state = None):
        return self.transform_q_function(q_function)
    