from ..core import Simulator, SystemState, SystemData

from stanza.dataclasses import dataclass

from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from typing import Sequence, Optional

import mujoco
import jax
import jax.numpy as jnp
import numpy as np
import os
import threading


# the mujoco simulator state is
# the SystemData and qacc_warmstart
@dataclass
class MujocoState:
    data: SystemData
    qacc_warmstart: jax.Array

# Used for the host <-> accelerator calls, the
# minimum state that needs to be passed to mimic simulator state
@dataclass
class MujocoStep:
    time: jax.Array
    qpos: jax.Array
    qvel: jax.Array
    act: jax.Array
    ctrl: jax.Array
    qacc_warmstart: jax.Array

class MujocoSimulator(Simulator[SystemData]):
    def __init__(self, model, threads=os.cpu_count()):
        # store up to 256 MjData objects for this model
        self.model = model
        example_data = mujoco.MjData(self.model)
        example_data = self._extract_data(example_data)
        self.data_structure = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape,
                x.dtype if x.dtype != np.float64 else np.float32
            ),
            example_data
        )
        self.buffer_width = self.model.vis.global_.offwidth
        self.buffer_height = self.model.vis.global_.offheight

        self.local_data = threading.local()
        # create an initial MjData object for each thread
        def initializer():
            self.local_data.renderer = None
            self.local_data.data = mujoco.MjData(self.model)
        self.pool = ThreadPoolExecutor(
            max_workers=threads, 
            initializer=initializer
        )

    @property
    def qpos0(self) -> jax.Array:
        return jnp.zeros_like(self.data_structure.qpos)

    @property
    def qvel0(self) -> jax.Array:
        return jnp.zeros_like(self.data_structure.qvel)

    @property
    def act0(self) -> jax.Array:
        return jnp.zeros_like(self.data_structure.act)
    
    @staticmethod
    def _extract_data(data: mujoco.MjData) -> SystemData:
        return SystemData(
            jnp.array(data.time, dtype=jnp.float32),
            jnp.copy(data.qpos.astype(jnp.float32)),
            jnp.copy(data.qvel.astype(jnp.float32)),
            jnp.copy(data.act.astype(jnp.float32)),
            jnp.copy(data.qacc.astype(jnp.float32)),
            jnp.copy(data.act_dot.astype(jnp.float32)),
            jnp.copy(data.xpos.astype(jnp.float32)),
            jnp.copy(data.xquat.astype(jnp.float32)),
            jnp.copy(data.site_xpos.astype(jnp.float32)),
            jnp.copy(data.site_xmat.astype(jnp.float32)),
            jnp.copy(data.actuator_velocity.astype(jnp.float32)),
            jnp.copy(data.cvel.astype(jnp.float32)),
            jnp.copy(data.qfrc_bias.astype(jnp.float32))
        )

    def _step_job(self, step: MujocoStep) -> MujocoState:
        # get the thread-local MjData object
        # copy over the jax arrays
        data = self.local_data.data
        data.time = step.time.item()
        data.qpos[:] = step.qpos
        data.qvel[:] = step.qvel
        data.act[:] = step.act
        data.ctrl[:] = step.ctrl
        data.qacc_warmstart[:] = step.qacc_warmstart
        mujoco.mj_step(self.model, data)
        state = MujocoState(
            data=self._extract_data(data),
            qacc_warmstart=jnp.copy(data.qacc_warmstart.astype(jnp.float32))
        )
        return state

    # (on host) step using the minimal amount of 
    # data that needs to be passed to the simulator
    def _step(self, step: MujocoStep) -> SystemData:
        if step.qpos.ndim == 1:
            data = self.pool.submit(self._step_job, step).result()
            return data
        else:
            assert False

    def step(self, state: MujocoState,
                   action : jax.Array, rng_key: jax.Array) -> SystemData:
        assert state.data.time.ndim == 0
        assert action.shape == (self.model.nu,), f"action shape {action.shape} != {self.model.nu}"
        return jax.pure_callback(
            self._step, MujocoState(
                self.data_structure,
                jax.ShapeDtypeStruct(
                    state.data.qvel.shape, state.data.qvel.dtype
                )
            ), 
            MujocoStep(
                state.data.time, state.data.qpos, state.data.qvel, 
                state.data.act, action, state.qacc_warmstart
            ),
            vectorized=False
        )

    def _forward_job(self, step: MujocoStep) -> MujocoState:
        data = self.local_data.data
        data.time = step.time.item()
        data.qpos[:] = step.qpos
        data.qvel[:] = step.qvel
        data.act[:] = step.act
        if step.ctrl is not None:
            data.ctrl[:] = np.zeros_like(step.ctrl)
        if step.qacc_warmstart is not None:
            data.qacc_warmstart[:] = step.qacc_warmstart
        mujoco.mj_forward(self.model, data)
        state = MujocoState(
            data=self._extract_data(data),
            qacc_warmstart=jnp.copy(data.qacc_warmstart.astype(jnp.float32))
        )
        return state

    # (on host) calls forward
    def _forward(self, step: MujocoStep) -> MujocoState:
        # regularize the batch shapes
        if step.qpos.ndim == 1: # if unvectorized
            data = self.pool.submit(self._forward_job, step).result()
            return data
        else:
            assert False
            # TODO: handle the vectorized case...
            # This is complicated as only certain
            # inputs may be vectorized

    def full_state(self, state: SystemState) -> MujocoState:
        assert state.time.ndim == 0
        step = MujocoStep(
            state.time, state.qpos, 
            state.qvel, state.act, None, None
        )
        structure = MujocoState(
            self.data_structure,
            jax.ShapeDtypeStruct(self.data_structure.qvel.shape, 
                                    self.data_structure.qvel.dtype)
        )
        return jax.pure_callback(self._forward, 
            structure, step, vectorized=False)

    def reduce_state(self, state: MujocoState) -> SystemState:
        return SystemState(
            state.data.time, state.data.qpos, state.data.qvel, state.data.act
        )

    def system_data(self, state: MujocoState) -> SystemData:
        return state.data
    
    def _get_jac_job(self, state: SystemState, id):
        data = self.local_data.data
        data.time = state.time
        data.qpos[:] = state.qpos
        data.qvel[:] = state.qvel
        data.act[:] = state.act
        mujoco.mj_forward(self.model, data)
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, data, jacp, None, id)
        mujoco.mj_jacSite(self.model, data, None, jacr, id)
        #jax.debug.print("jacp: {s}", s=jacp)
        return jacp.astype(jnp.float32), jacr.astype(jnp.float32)
    
    def _get_jac(self, state, id):
        return self.pool.submit(self._get_jac_job, state, id).result()
    
    def get_jacs(self, state: SystemState, id: int) -> jax.Array:
        """Returns the position and orientation parts of the Jacobian of the site at the given id."""
        structure = (jnp.zeros((3, self.model.nv), dtype=jnp.float32), jnp.zeros((3, self.model.nv), dtype=jnp.float32))
        jacp, jacr = jax.pure_callback(self._get_jac, structure, state, id)
        return jacp, jacr
    
    def _get_fullM_job(self, state: SystemState):
        data = self.local_data.data
        data.time = state.time
        data.qpos[:] = state.qpos
        data.qvel[:] = state.qvel
        data.act[:] = state.act
        mujoco.mj_forward(self.model, data)
        mass_matrix = np.zeros((self.model.nv, self.model.nv), dtype=np.float64, order="C")
        mujoco.mj_fullM(self.model, mass_matrix, np.array(data.qM))
        return mass_matrix.astype(jnp.float32)
    
    def _get_fullM(self, state: SystemState):
        return self.pool.submit(self._get_fullM_job, state).result()

    def get_fullM(self, state: SystemState) -> jax.Array:
        """Returns the full mass matrix."""
        structure = jnp.zeros((self.model.nv, self.model.nv), dtype=jnp.float32)
        M = jax.pure_callback(self._get_fullM, structure, state)
        M = M.reshape((self.model.nv, self.model.nv))
        return M
    

    # render a given SystemData using
    # the opengl-based mujoco rendering engine

    def _render_job(self, width : int, height: int,
                    geom_groups : Sequence[bool],
                    camera : int | str,
                    state: SystemState,
                    trajectory: Optional[jax.Array] = None) -> jax.Array:
        # get the thread-local MjData object
        # copy over the jax arrays
        renderer = self.local_data.renderer
        if renderer is None:
            renderer = mujoco.Renderer(
                self.model, self.buffer_height, self.buffer_width
            )
            self.local_data.renderer = renderer
        ldata = self.local_data.data
        ldata.time = 0
        ldata.qpos[:] = state.qpos
        ldata.qvel[:] = state.qvel
        mujoco.mj_forward(self.model, ldata)
        vopt = mujoco.MjvOption()
        # disable rendering of collision geoms
        for i, g in enumerate(geom_groups):
            vopt.geomgroup[i] = 1 if g else 0
        arr = np.empty((self.buffer_height, self.buffer_width, 3), dtype=np.uint8)
        renderer.update_scene(ldata, camera, vopt)
        if trajectory is not None:
            def addSphere(scene, pos1, pos2, rgba):
                if scene.ngeom >= scene.maxgeom:
                    return
                scene.ngeom += 1  # increment ngeom
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                                    mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                                    np.zeros(3), np.zeros(9), rgba)
                mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                           mujoco.mjtGeom.mjGEOM_CAPSULE, 0.01,
                           pos1[0], pos1[1], pos1[2],
                           pos2[0], pos2[1], pos2[2])
            T = trajectory.shape[0]
            colors = np.array((np.arange(T)/T, np.zeros(T), np.ones(T), np.ones(T)), dtype=np.float32).T
            for i in range(trajectory.shape[0]-1):
                addSphere(renderer.scene, trajectory[i], trajectory[i+1], colors[i])
        renderer.render(out=arr)
        return jnp.array(arr)

    def _render(self, width, height, geom_groups, camera, state: SystemState, 
                trajectory: Optional[jax.Array] = None) -> jax.Array:
        job = partial(self._render_job, width, height, geom_groups, camera)
        if state.qpos.ndim == 1:
            data = self.pool.submit(job, state, trajectory).result()
            return data
        else:
            assert False

    def render(self, state: SystemState, width: int, height: int, geom_groups: Sequence[bool], camera: int | str = -1,
               trajectory: Optional[jax.Array] = None) -> jax.Array:
        assert state.time.ndim == 0
        buffer = jax.pure_callback(
            partial(self._render, width, height, geom_groups, camera),
            jax.ShapeDtypeStruct((self.buffer_height, self.buffer_width, 3), jnp.uint8),
            state, trajectory, vectorized=False
        )
        buffer = buffer.astype(jnp.float32) / 255.0
        buffer = jax.image.resize(buffer, (height, width,3), method="linear")
        return buffer



jax.tree_util.register_static(MujocoSimulator)