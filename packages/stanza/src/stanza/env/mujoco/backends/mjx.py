import jax
import jax.numpy as jnp
import jax.experimental

from mujoco import mjx
from stanza.env.mujoco.core import SystemState

from ..core import Simulator, SystemState, SystemData

class MjxSimulator(Simulator[mjx.Data]):
    def __init__(self, model):
        if isinstance(model, mjx.Model):
            self.model = model
        else:
            with jax.experimental.disable_x64():
                self.model = mjx.put_model(model)

    @property
    def qpos0(self) -> jax.Array:
        return jnp.zeros_like(self.model.qpos0)

    @property
    def qvel0(self) -> jax.Array:
        return jnp.zeros_like(self.model.qvel0)

    @property
    def act0(self) -> jax.Array:
        return jnp.zeros_like(self.model.act0)

    @jax.jit
    def step(self, state, action, rng_key):
        if action is not None:
            state = state.replace(ctrl=action)
        with jax.experimental.disable_x64():
            state = mjx.step(self.model, state)
        state = jax.tree.map(
            lambda x: x.astype(jnp.float32) if x.dtype == jnp.float64 else x,
            state
        )
        return state
    
    @jax.jit
    def full_state(self, state: SystemState) -> mjx.Data:
        with jax.experimental.disable_x64():
            data = mjx.make_data(self.model)
            data = data.replace(time=state.time, 
                        qpos=state.qpos,
                        qvel=state.qvel,
                        act=state.act)
            data = mjx.forward(self.model, data)
        data = jax.tree.map(
            lambda x: x.astype(jnp.float32) if x.dtype == jnp.float64 else x,
            data
        )
        return data
    
    def reduce_state(self, state: mjx.Data) -> SystemState:
        return SystemState(
            time=state.time,
            qpos=state.qpos,
            qvel=state.qvel,
            act=state.act
        )

    def system_data(self, state: mjx.Data) -> SystemData:
        return SystemData(
            time=state.time,
            qpos=state.qpos,
            qvel=state.qvel,
            act=state.act,
            qacc=state.qacc,
            act_dot=state.act_dot,
            xpos=state.xpos,
            xquat=state.xquat,
            site_xpos=state.site_xpos,
            site_xmat=None,
            actuator_velocity=state.actuator_velocity,
            cvel=state.cvel,
            qfrc_bias=state.qfrc_bias
        )

jax.tree_util.register_pytree_node(
    MjxSimulator,
    lambda x: ([x.model,], None),
    lambda aux, x: MjxSimulator(x[0])
)


import mujoco.mjx._src.math

def axis_angle_to_quat(axis: jax.Array, angle: jax.Array) -> jax.Array:
  """Provides a quaternion that describes rotating around axis by angle.

  Args:
    axis: (3,) axis (x,y,z)
    angle: () float angle to rotate by

  Returns:
    A quaternion that rotates around axis by angle
  """
  s, c = jnp.sin(angle * 0.5), jnp.cos(angle * 0.5)
  return jnp.concatenate((c[None], axis * s), axis=0)

# Patch to fix x64 issue with jnp.insert()
mujoco.mjx._src.math.axis_angle_to_quat = axis_angle_to_quat