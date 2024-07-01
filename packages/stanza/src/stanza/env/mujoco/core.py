from stanza.env import Environment
from stanza.dataclasses import dataclass, field
from stanza.util import jax_static_property

from stanza import canvas

from typing import TypeVar, Generic

import mujoco
import numpy as np

import jax
import jax.numpy as jnp
import abc

SimulatorState = TypeVar("SimulatorState")
Action = jax.Array

# The physics simulator abstraction:
@dataclass
class SystemState:
    time: jax.Array
    qpos: jax.Array
    qvel: jax.Array
    act: jax.Array # actuator state

# The system data
# including observables
@dataclass
class SystemData(SystemState):
    # dynamics:
    qacc: jax.Array
    act_dot: jax.Array
    # position dependent:
    xpos: jax.Array
    xquat: jax.Array
    # position, velocity dependent:
    actuator_velocity: jax.Array
    cvel: jax.Array

# Must be a jax pytree type!
class Simulator(abc.ABC, Generic[SimulatorState]):
    @abc.abstractmethod
    def step(self, state: SimulatorState, 
            action : Action, rng_key : jax.Array) -> SimulatorState: ...

    @abc.abstractmethod
    def data(self, state: SimulatorState) -> SystemData: ...

    @abc.abstractmethod
    def reduce_state(self, state: SimulatorState) -> SystemState: ...

    @abc.abstractmethod
    def full_state(self, state: SystemState) -> SimulatorState: ...

@dataclass(kw_only=True)
class MujocoEnvironment(Environment[SimulatorState, SystemState, Action], Generic[SimulatorState]):
    physics_backend : str = field(default="mujoco", pytree_node=False)

    # Implement "xml"
    @jax_static_property
    def xml(self):
        raise NotImplementedError()

    @jax_static_property
    def model(self):
        return mujoco.MjModel.from_xml_string(self.xml)
    
    @jax_static_property
    def simulator(self):
        return self.make_simulator(self.physics_backend, self.model)
    
    def full_state(self, reduced_state: SystemState) -> SimulatorState:
        return self.simulator.full_state(reduced_state)
    
    def reduce_state(self, full_state: SimulatorState) -> SystemState:
        return self.simulator.reduce_state(full_state)

    @jax.jit
    def reset(self, rng_key: jax.Array) -> SimulatorState:
        raise NotImplementedError()

    @jax.jit
    def step(self, state: SimulatorState, action: Action, 
                    rng_key: jax.Array) -> SimulatorState:
        return self.simulator.step(state, action, rng_key)

    # Creates the backend for this environment

    @staticmethod
    def make_simulator(backend : str, model: mujoco.MjModel) -> Simulator:
        if backend == "mujoco":
            from .backends.mujoco import MujocoSimulator
            return MujocoSimulator(model)
        elif backend == "mjx":
            from .backends.mjx import MjxSimulator
            return MjxSimulator(model)
        elif backend == "brax":
            from .backends.brax import BraxSimulator
            return BraxSimulator(model)

# Utilities
def render_2d(model: mujoco.MjModel, data: SystemData, 
                width, height, 
                world_width, world_height,
                # render a subset of the bodies
                # in the specified colors:
                body_custom=None):
    geoms = []
    for type, size, body, pos, quat, rgba in zip(
            model.geom_type, model.geom_size,
            model.geom_bodyid, model.geom_pos,
            model.geom_quat, model.geom_rgba):
        body_pos = data.xpos[body][:2]
        body_rot = quat_to_angle(data.xquat[body])

        # override the body position and rotation
        if body_custom is not None and body not in body_custom:
            continue
        elif body_custom is not None and body in body_custom:
            body_pos, body_rot, rgba = body_custom[body]

        if type == mujoco.mjtGeom.mjGEOM_BOX:
            geom = canvas.box((-size[0], -size[1]), (size[0], size[1]))
        elif (type == mujoco.mjtGeom.mjGEOM_CYLINDER \
                or type == mujoco.mjtGeom.mjGEOM_SPHERE):
            geom = canvas.circle((0,0), size[0])
        elif type == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        else:
            raise ValueError(f"Unsupported geometry type {type}!")
        geom = canvas.fill(geom, rgba[:3])
        # get the body position from the state
        geom = canvas.transform(geom, 
            translation=pos[:2],
            rotation=quat_to_angle(quat))
        geom = canvas.transform(geom, 
            translation=body_pos,
            rotation=body_rot)
        geoms.append(geom)
    world = canvas.stack(*geoms)
    world = canvas.transform(world, scale=(1, -1))
    world = canvas.transform(
        world, translation=(world_width/2, world_height/2),
        scale=(width/world_width, height/world_height)
    )
    return world

def get_custom_data(model : mujoco.MjModel):
    custom = {}
    for name_adr, data_adr, data_len in zip(
                model.name_numericadr,
                model.numeric_adr,
                model.numeric_data):
        data_adr, data_len = int(data_adr), int(data_len)
        name_len = model.names[name_adr:].index(0)
        name = model.names[name_adr:name_adr+name_len].decode("utf-8")
        data = model.numeric_data[data_adr:data_adr + data_len]
        custom[name] = data
    return custom

def quat_to_angle(quat):
    w0 = quat[0] # cos(theta/2)
    w3 = quat[3] # sin(theta/2)
    angle = 2*jax.numpy.atan2(w3, w0)
    return angle