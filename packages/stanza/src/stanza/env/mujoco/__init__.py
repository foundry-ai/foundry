from stanza.env import EnvironmentRegistry, Environment
from stanza.util.registry import from_module

from functools import cached_property
from stanza.dataclasses import dataclass, field

import jax
import mujoco
from mujoco import mjx


import brax
import brax.io.mjcf
import brax.io.html
import brax.mjx.pipeline as pipeline
from brax.base import Contact, Motion, System, Transform

@dataclass
class MujocoState:
    q: jax.Array
    qd: jax.Array

@dataclass
class MujocoEnvironment(Environment):
    backend: str = field(default="mjx", pytree_node=False) # (mjx, brax, mujoco)

    @property
    def xml(self):
        pass

    @cached_property
    def mj_model(self): 
        """Load and return the mujoco mjModel."""
        with jax.ensure_compile_time_eval():
            return mujoco.MjModel.from_xml_string(self.xml)

    @cached_property
    def mjx_model(self):
        """Load and return the MJX model."""
        with jax.ensure_compile_time_eval():
            model = self.mj_model
            return mjx.put_model(model)
    
    @staticmethod
    def _quat_to_angle(quat):
        w0 = quat[0] # cos(theta/2)
        w3 = quat[3] # sin(theta/2)
        angle = 2*jax.numpy.atan2(w3, w0)
        return angle
    
    @jax.jit
    def step(self, state, action, rng_key = None): 
        data = mjx.make_data(self.mjx_model)
        data = data.replace(qpos=state.q, qvel=state.qd)
        if action is not None:
            data = data.replace(ctrl=action)
        @jax.jit
        def step_fn_mjx(data, _):
            return mjx.step(self.mjx_model, data), None
        def step_fn_mujoco(data, _):
            return mujoco.mj_step(self.mj_model, data), None
        if self.backend == "mjx":
            data, _ = jax.lax.scan(step_fn_mjx, data, length=6)
        elif self.backend == "mujoco":
            data, _ = jax.lax.scan(step_fn_mujoco, data, length=6)
        return MujocoState(data.qpos, data.qvel)

    @staticmethod
    def brax_to_state(sys, data):
        q, qd = data.qpos, data.qvel
        x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
        cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
        offset = data.xpos[1:, :] - data.subtree_com[sys.body_rootid[1:]]
        offset = Transform.create(pos=offset)
        xd = offset.vmap().do(cvel)
        data = pipeline._reformat_contact(sys, data)
        return pipeline.State(q=q, qd=qd, x=x, xd=xd, **data.__dict__)

    @staticmethod
    def brax_render(mj_model, data_seq):
        sys = brax.io.mjcf.load_model(mj_model)
        T = data_seq.xpos.shape[0]
        states = jax.vmap(MujocoEnvironment.brax_to_state, in_axes=(None, 0))(sys, data_seq)
        states = [jax.tree_map(lambda x: x[i], states) for i in range(T)]
        return brax.io.html.render(sys, states)

environments = EnvironmentRegistry[Environment]()
environments.extend("pusht", from_module(".pusht", "environments"))