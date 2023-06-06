from stanza.envs.gymnasium import GymWrapper
from stanza.util.dataclasses import dataclass, field, replace
from stanza.util.attrdict import AttrMap
from typing import Callable

import jax.numpy as jnp

@dataclass(jax=True)
class BodyState:
    position: jnp.array
    velocity: jnp.array
    angle: jnp.array
    angular_velocity: jnp.array

class SystemState(AttrMap):
    pass

class SystemDef(AttrMap):
    pass

class PyMunkGym:
    pass

@dataclass(jax=True, kw_only=True)
class PyMunkWrapper(GymWrapper):
    sim_hz : float = 100

    def _make_system(self, rng_key):
        pass

    def _make_env(self, rng_key):
        system = self.system_builder()
        return PyMunkGym(system, self.sim_hz)

@dataclass(jax=True, kw_only=True)
class PyMunkEnv(PyMunkWrapper):
    def _make_system(self, rng_key):
        pass