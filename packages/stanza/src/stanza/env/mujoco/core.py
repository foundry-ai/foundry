from stanza.env import (
    Environment, RenderConfig,
    HtmlRender, ImageRender
)
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

    # site locations
    site_xpos: jax.Array
    site_xmat: jax.Array

    # position, velocity dependent:
    actuator_velocity: jax.Array
    cvel: jax.Array
    # forces
    qfrc_bias: jax.Array

# Must be a jax pytree type!
class Simulator(abc.ABC, Generic[SimulatorState]):
    @property
    def qpos0(self) -> jax.Array: ...

    @abc.abstractmethod
    def step(self, state: SimulatorState, 
            action : Action, rng_key : jax.Array) -> SimulatorState: ...

    @abc.abstractmethod
    def system_data(self, state: SimulatorState) -> SystemData: ...

    @abc.abstractmethod
    def reduce_state(self, state: SimulatorState) -> SystemState: ...

    @abc.abstractmethod
    def full_state(self, state: SystemState) -> SimulatorState: ...

@dataclass(kw_only=True)
class MujocoEnvironment(Environment[SimulatorState, SystemState, Action], Generic[SimulatorState]):
    physics_backend : str = field(default="mujoco", pytree_node=False)

    @jax_static_property
    def model(self) -> mujoco.MjModel:
        raise NotImplementedError()
    
    @jax_static_property
    def simulator(self) -> Simulator:
        return self.make_simulator(self.physics_backend, self.model)

    # Always guaranteed to be a MujocoSimulator,
    # even if the physics backend is not "mujoco."
    # This is used as a fallback for rendering.
    @jax_static_property
    def native_simulator(self) -> Simulator:
        if self.physics_backend == "mujoco":
            return self.simulator
        from .backends.mujoco import MujocoSimulator
        return MujocoSimulator(self.model)

    def full_state(self, reduced_state: SystemState) -> SimulatorState:
        return self.simulator.full_state(reduced_state)
    
    def reduce_state(self, full_state: SimulatorState) -> SystemState:
        return self.simulator.reduce_state(full_state)

    @jax.jit
    def sample_action(self, rng_key):
        return jnp.zeros((self.simulator.model.nu,), jnp.float32)

    @jax.jit
    def sample_state(self, rng_key):
        return self.full_state(SystemState(
            time=jnp.zeros((), jnp.float32),
            qpos=self.simulator.qpos0,
            qvel=self.simulator.qvel0,
            act=self.simulator.act0
        ))

    @jax.jit
    def reset(self, rng_key: jax.Array) -> SimulatorState:
        raise NotImplementedError()

    @jax.jit
    def step(self, state: SimulatorState, action: Action, 
                    rng_key: jax.Array) -> SimulatorState:
        return self.simulator.step(state, action, rng_key)

    @jax.jit
    def render(self, state: SimulatorState, config: RenderConfig | None = None) -> jax.Array:
        config = config or ImageRender(width=256, height=256)
        if isinstance(config, ImageRender):
            state = self.simulator.reduce_state(state)
            camera = config.camera if config.camera is not None else -1
            return self.native_simulator.render(
                state, config.width, config.height, (), camera, config.trajectory
            )
        elif isinstance(config, HtmlRender):
            data = self.simulator.system_data(state) # type: SystemData
            if data.qpos.ndim == 1:
                data = jax.tree_map(lambda x: jnp.expand_dims(x, 0), data)
        else:
            raise ValueError("Unsupported render config")

    # Creates the backend for this environment

    @staticmethod
    def make_simulator(backend : str, model: mujoco.MjModel) -> Simulator:
        if backend == "mujoco":
            from .backends.mujoco import MujocoSimulator
            return MujocoSimulator(model)
        elif backend == "mjx":
            from .backends.mjx import MjxSimulator
            return MjxSimulator(model)
        else:
            raise ValueError(f"Unsupported simulator backend {backend}")

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

def quat_to_mat(quat):
    """
    Adapted from diffusion_policy quatmath.py.
    Converts given quaternion to matrix.

    Args:
        quat (jnp.array): (x,y,z,w) vec4 float angles

    Returns:
        jnp.array: 3x3 rotation matrix
    """
    quat = jnp.asarray(quat, dtype=jnp.float32)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = jnp.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = jnp.empty(quat.shape[:-1] + (3, 3), dtype=jnp.float32)
    mat = mat.at[..., 0, 0].set(1.0 - (yY + zZ))
    mat = mat.at[..., 0, 1].set(xY - wZ)
    mat = mat.at[..., 0, 2].set(xZ + wY)
    mat = mat.at[..., 1, 0].set(xY + wZ)
    mat = mat.at[..., 1, 1].set(1.0 - (xX + zZ))
    mat = mat.at[..., 1, 2].set(yZ - wX)
    mat = mat.at[..., 2, 0].set(xZ - wY)
    mat = mat.at[..., 2, 1].set(yZ + wX)
    mat = mat.at[..., 2, 2].set(1.0 - (xX + yY))
    return jnp.where((Nq > 1e-6)[..., jnp.newaxis, jnp.newaxis], mat, jnp.eye(3))

def mat_to_quat(rmat):
    """
    Adapted from robosuite transform_utils.py
    Converts given rotation matrix to quaternion.

    Args:
        rmat (jnp.array): 3x3 rotation matrix

    Returns:
        jnp.array: (x,y,z,w) float quaternion angles
    """
    M = jnp.asarray(rmat).astype(jnp.float32)[:3, :3]

    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = jnp.array(
        [
            [m00 - m11 - m22, jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, jnp.float32(0.0), jnp.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, jnp.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = jnp.linalg.eigh(K)
    inds = jnp.array([3, 0, 1, 2])
    q1 = V[inds, jnp.argmax(w)]
    q1 *= jnp.sign(q1[0])
    inds = jnp.array([1, 2, 3, 0])
    return q1[inds]

def mat_to_euler(mat):
    """ 
    Adapted from diffusion_policy quatmath.py. 
    Convert Rotation Matrix to Euler Angles. 
    """
    mat = jnp.asarray(mat, dtype=jnp.float32)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = jnp.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > 1e-6
    euler = jnp.empty(mat.shape[:-1], dtype=jnp.float32)
    euler = euler.at[..., 2].set(jnp.where(condition,
                             -jnp.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                             -jnp.arctan2(-mat[..., 1, 0], mat[..., 1, 1])))
    euler = euler.at[..., 1].set(jnp.where(condition,
                             -jnp.arctan2(-mat[..., 0, 2], cy),
                             -jnp.arctan2(-mat[..., 0, 2], cy)))
    euler = euler.at[..., 0].set(jnp.where(condition,
                             -jnp.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
                             0.0))
    return euler

def orientation_error(desired, current):
    """
    Adapted from robosuite control_utils.py
    This function calculates a 3-dimensional orientation error vector for use in the
    impedance controller. It does this by computing the delta rotation between the
    inputs and converting that rotation to exponential coordinates (axis-angle
    representation, where the 3d vector is axis * angle).
    See https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation for more information.
    Optimized function to determine orientation error from matrices

    Args:
        desired (np.array): 2d array representing target orientation matrix
        current (np.array): 2d array representing current orientation matrix

    Returns:
        np.array: 2d array representing orientation error as a matrix
    """
    rc1 = current[0:3, 0]
    rc2 = current[0:3, 1]
    rc3 = current[0:3, 2]
    rd1 = desired[0:3, 0]
    rd2 = desired[0:3, 1]
    rd3 = desired[0:3, 2]

    error = 0.5 * (jnp.cross(rc1, rd1) + jnp.cross(rc2, rd2) + jnp.cross(rc3, rd3))

    return error



