from ..core import Simulator, SystemState, SystemData

from foundry.core.dataclasses import dataclass
from foundry.core import tree

from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Sequence, Optional

import mujoco
import jax
import foundry.numpy as jnp
import math
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
    ctrl: jax.Array | None
    qacc_warmstart: jax.Array | None

class MujocoSimulator(Simulator[SystemData]):
    def __init__(self, model, dtype=jnp.float32, threads=2*os.cpu_count()):
        # store up to 256 MjData objects for this model
        self.model = model
        self.dtype = dtype

        example_mjdata = mujoco.MjData(self.model)
        example_data = self._extract_data(example_mjdata)
        self.data_structure = tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape,
                x.dtype if x.dtype != np.float64 else np.float32
            ),
            example_data
        )
        self.state_structure = MujocoState(
            self.data_structure,
            jax.ShapeDtypeStruct(example_mjdata.qacc_warmstart.shape, dtype)
        )
        self.step_structure = MujocoStep(
            jax.ShapeDtypeStruct((), dtype),
            jax.ShapeDtypeStruct(example_data.qpos.shape, dtype),
            jax.ShapeDtypeStruct(example_data.qvel.shape, dtype),
            jax.ShapeDtypeStruct(example_data.act.shape, dtype),
            jax.ShapeDtypeStruct(example_mjdata.ctrl.shape, dtype),
            jax.ShapeDtypeStruct(example_data.qacc.shape, dtype)
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
    
    def _extract_data(self, data: mujoco.MjData) -> SystemData:
        return SystemData(
            np.array(data.time, dtype=self.dtype),
            np.copy(data.qpos.astype(self.dtype)),
            np.copy(data.qvel.astype(self.dtype)),
            np.copy(data.act.astype(self.dtype)),
            np.copy(data.qacc.astype(self.dtype)),
            np.copy(data.act_dot.astype(self.dtype)),
            np.copy(data.xpos.astype(self.dtype)),
            np.copy(data.xquat.astype(self.dtype)),
            np.copy(data.site_xpos.astype(self.dtype)),
            np.copy(data.site_xmat.astype(self.dtype)),
            np.copy(data.actuator_velocity.astype(self.dtype)),
            np.copy(data.cvel.astype(self.dtype)),
            np.copy(data.qfrc_bias.astype(self.dtype))
        )

    def _run_job(self, func, step: MujocoStep, *args): 
        # get the thread-local MjData object
        # copy over the jax arrays
        data = self.local_data.data
        data.time = step.time.item()
        data.qpos[:] = step.qpos
        data.qvel[:] = step.qvel
        data.act[:] = step.act
        if step.ctrl is not None:
            data.ctrl[:] = step.ctrl
        if step.qacc_warmstart is not None:
            data.qacc_warmstart[:] = step.qacc_warmstart
        return func(data, *args)

    def _job_callback(self, func, *args):
        def run_job(args):
            return self._run_job(func, *args)
        batch_shape = args[0].time.shape
        batch_dims = len(batch_shape)
        N = math.prod(batch_shape)

        # reshape the args
        args = tree.map(
            lambda x: np.array(x).reshape((N,) + x.shape[batch_dims:]),  args
        )
        results = self.pool.map(run_job, (tree.map(lambda x: x[i], args) for i in range(N)))
        # stack everything
        results = tree.map(lambda *x: np.stack(x, axis=0), *results)
        # reshape to non-batch
        results = tree.map(lambda x: x.reshape(batch_shape + x.shape[1:]), results)
        return results

    def _forward_job(self, data : mujoco.MjData) -> MujocoState:
        mujoco.mj_forward(self.model, data)
        return MujocoState(
            data=self._extract_data(data),
            qacc_warmstart=np.copy(data.qacc_warmstart.astype(jnp.float32))
        )


    def full_state(self, state: SystemState) -> MujocoState:
        assert state.time.ndim == 0
        step = MujocoStep(
            state.time, state.qpos, 
            state.qvel, state.act, None, None
        )
        return jax.pure_callback(
            partial(self._job_callback, self._forward_job), 
            self.state_structure, step, vmap_method="broadcast_all"
        )

    def _step_job(self, data : mujoco.MjData) -> MujocoState:
        mujoco.mj_step(self.model, data)
        return MujocoState(
            data=self._extract_data(data),
            qacc_warmstart=np.copy(data.qacc_warmstart.astype(self.dtype))
        )

    def step(self, state: MujocoState,
                   action : jax.Array, rng_key: jax.Array) -> SystemData:
        assert state.data.time.ndim == 0, "time must be unbatched"
        assert action.shape == (self.model.nu,), f"action shape {action.shape} != {self.model.nu}"

        step = MujocoStep(
            state.data.time, state.data.qpos, state.data.qvel, 
            state.data.act, action, state.qacc_warmstart
        )
        return jax.pure_callback(
            partial(self._job_callback, self._step_job),
            self.state_structure, step, vmap_method="broadcast_all"
        )

    def reduce_state(self, state: MujocoState) -> SystemState:
        return SystemState(
            state.data.time, state.data.qpos, state.data.qvel, state.data.act
        )

    def system_data(self, state: MujocoState) -> SystemData:
        return state.data
    
    def _fullM_job(self, data : mujoco.MjData) -> jax.Array:
        mujoco.mj_forward(self.model, data)
        mass_matrix = np.zeros((self.model.nv, self.model.nv), dtype=np.float64, order="C")
        mujoco.mj_fullM(self.model, mass_matrix, np.array(data.qM))
        return mass_matrix.astype(self.dtype)

    def get_fullM(self, state: SystemState) -> jax.Array:
        """Returns the full mass matrix."""
        structure = jnp.zeros((self.model.nv, self.model.nv), dtype=jnp.float32)
        step = MujocoStep(state.time, state.qpos, state.qvel, state.act, None, None)
        M = jax.pure_callback(
            partial(self._job_callback, self._fullM_job),
            structure, step, vmap_method="broadcast_all"
        )
        return M

    def _jac_job(self, data : mujoco.MjData, site_id : jax.Array) -> tuple[jax.Array, jax.Array]:
        site_id = site_id.item()
        mujoco.mj_forward(self.model, data)
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, data, jacp, None, site_id)
        mujoco.mj_jacSite(self.model, data, None, jacr, site_id)
        return jacp.astype(self.dtype), jacr.astype(self.dtype)
    
    def get_jacs(self, state: SystemState, id: int) -> jax.Array:
        """Returns the position and orientation parts of the Jacobian of the site at the given id."""
        structure = (jax.ShapeDtypeStruct((3, self.model.nv), self.dtype), 
                     jax.ShapeDtypeStruct((3, self.model.nv), self.dtype))
        step = MujocoStep(state.time, state.qpos, state.qvel, state.act, None, None)
        jacp, jacr = jax.pure_callback(
            partial(self._job_callback, self._jac_job),
            structure, step, id, vmap_method="broadcast_all"
        )
        return jacp, jacr

    # render a given SystemData using
    # the opengl-based mujoco rendering engine
    def _render_job(self, width : int, height: int,
                    geom_groups : Sequence[bool],
                    camera : int | str,
                    data: mujoco.MjData,
                    trajectory: Optional[jax.Array] = None) -> jax.Array:
        # get the thread-local MjData object
        # copy over the jax arrays
        renderer = self.local_data.renderer
        if renderer is None:
            renderer = mujoco.Renderer(
                self.model, self.buffer_height, self.buffer_width
            )
            self.local_data.renderer = renderer
        mujoco.mj_forward(self.model, data)
        vopt = mujoco.MjvOption()
        # disable rendering of collision geoms
        for i, g in enumerate(geom_groups):
            vopt.geomgroup[i] = 1 if g else 0
        arr = np.empty((self.buffer_height, self.buffer_width, 3), dtype=np.uint8)
        renderer.update_scene(data, camera, vopt)
        if trajectory is not None:
            def addSphere(scene, pos1, pos2, rgba):
                if scene.ngeom >= scene.maxgeom:
                    return
                scene.ngeom += 1  # increment ngeom
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                                    mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                                    np.zeros(3), np.zeros(9), rgba)
                mujoco.mjv_connector(scene.geoms[scene.ngeom-1],
                           mujoco.mjtGeom.mjGEOM_CAPSULE, 0.01,
                           pos1, pos2)
            T = trajectory.shape[0]
            colors = np.array((np.arange(T)/T, np.zeros(T), np.ones(T), np.ones(T)), dtype=np.float32).T
            for i in range(trajectory.shape[0]-1):
                addSphere(renderer.scene, trajectory[i, :3].astype(np.float64), trajectory[i+1, :3].astype(np.float64), colors[i])
        renderer.render(out=arr)
        return jnp.array(arr)

    def render(self, state: SystemState, width: int, height: int, geom_groups: Sequence[bool], camera: int | str = -1,
               trajectory: Optional[jax.Array] = None) -> jax.Array:
        assert state.time.ndim == 0
        render_job = partial(self._render_job, width, height, geom_groups, camera)

        step = MujocoStep(
            state.time, state.qpos, 
            state.qvel, state.act, None, None
        )
        trajectory_structure = jax.ShapeDtypeStruct(trajectory.shape, jnp.float32) if trajectory is not None else None
        buffer = jax.pure_callback(
            partial(self._job_callback, render_job),
            jax.ShapeDtypeStruct((self.buffer_height, self.buffer_width, 3), jnp.uint8),
            step, trajectory, vmap_method="broadcast_all"
        )
        buffer = buffer.astype(jnp.float32) / 255.0
        buffer = jax.image.resize(buffer, (height, width,3), method="linear")
        return buffer

jax.tree_util.register_static(MujocoSimulator)