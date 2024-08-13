import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import collections.abc

from stanza.dataclasses import dataclass, field, replace
from stanza.util import jax_static_property
from stanza.env import (
    EnvWrapper, RenderConfig, ObserveConfig,
    ImageRender, SequenceRender,
    Environment,
    EnvironmentRegistry
)
from stanza.env.transforms import (
    EnvTransform, ChainedTransform,
    MultiStepTransform
)
from stanza.env.mujoco.core import (
    MujocoEnvironment,
    SystemState, SimulatorState, Action,
    quat_to_mat, orientation_error
)

from functools import partial
from typing import Sequence, Callable, Iterable, Any

# handles the XML generation, rendering

ROBOT_NAME_MAP = {"panda": "Panda"}

# our initializer
RobotInitializer = Any
ObjectInitializer = Any

@dataclass(kw_only=True)
class RobosuiteEnv(MujocoEnvironment[SimulatorState]):
    robots: Sequence[str] = field(pytree_node=False)

    @property
    def _env_args(self):
        return {}

    @jax_static_property
    def _model_initializers(self) -> tuple[mujoco.MjModel, Sequence[RobotInitializer], ObjectInitializer]:
        _setup_macros()
        import robosuite as suite
        env = suite.make(
            env_name=self.env_name,
            robots=[ROBOT_NAME_MAP[r] for r in self.robots],
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            ignore_done=True,
            **self._env_args
        )
        env.reset()
        model =  env.sim.model._model
        robots = tuple(RobotInitializer.from_robosuite(r) for r in env.robots)
        initializer = ObjectInitializer.from_robosuite(env.placement_initializer)
        return model, robots, initializer
    
    @property
    def model(self):
        return self._model_initializers[0]

    @jax.jit
    def _reset_internal(self, rng_key : jax.Array) -> SystemState:
        state = SystemState(
            time=jnp.zeros((), jnp.float32),
            qpos=self.simulator.qpos0,
            qvel=self.simulator.qvel0,
            act=self.simulator.act0
        )
        # get the model, robot initializers, and object initializers
        model, robots, initializer = self._model_initializers
        keys = jax.random.split(rng_key, len(robots) + 1)
        r_keys, i_key = keys[:-1], keys[-1]
        for rk, r in zip(r_keys, robots):
            state = r.reset(rk, state)
        placements = initializer.generate(i_key)
        state = ObjectInitializer.update_state(model, state, placements.values())
        return state
    
    @jax.jit
    def reset(self, rng_key: jax.Array) -> SimulatorState:
        state = self._reset_internal(rng_key)
        return self.full_state(state)

    # use the "frontview" camera for rendering
    # if not specified
    def render(self, state, config = None):
        config = config or ImageRender(width=256, height=256)
        # custom image rendering for robosuite
        # which disables the collision geometry visualization
        if isinstance(config, ImageRender):
            state = self.simulator.reduce_state(state)
            camera = config.camera if config.camera is not None else "frontview"
            # render only the visual geometries
            # do not include the collision geometries
            return self.native_simulator.render(
                state, config.width, config.height, (False, True), camera
            )
        return super().render(config, state)

_OBJECT_JOINT_MAP = {
    "can": "Can_joint0",
    "milk": "Milk_joint0",
    "bread": "Bread_joint0",
    "cereal": "Cereal_joint0"
}
_TARGET_BIN_ID = {
    "milk": 0,
    "bread": 1,
    "cereal": 2,
    "can": 3
}

@dataclass(kw_only=True)
class PickAndPlace(RobosuiteEnv[SimulatorState]):
    num_objects: int = field(default=4, pytree_node=False)
    objects: Sequence[str] = field(
        default=("can","milk","bread","cereal"), 
        pytree_node=False
    )

    @property
    def env_name(self):
        return "PickPlace"

    @property
    def _env_args(self):
        if self.num_objects == 1 and len(self.objects) == 1:
            return {"single_object_mode": 2, "object_type": self.objects[0]}
        elif self.num_objects == 1:
            return {"single_object_mode": 1, "object_type": self.objects[0]}
        else:
            return {"single_object_mode": 0}

    def _reset_internal(self, rng_key : jax.Array) -> SystemState:
        # randomly select one of the nuts to remove
        def remove_objects(state, names):
            qpos = state.qpos
            for n in names:
                qpos = _set_joint_qpos(self.model, _OBJECT_JOINT_MAP[n],
                    qpos, jnp.array([0, 0, 0, 1, 0, 0, 0])
                )
            return replace(state, qpos=qpos)

        if self.num_objects == 1 and len(self.objects) > 1:
            # randomly select objects to remove
            objects = list(self.objects)
            rng_key, sk = jax.random.split(rng_key)
            state = super()._reset_internal(rng_key)
            branches = [partial(remove_objects, names=[o for o in objects if o != obj]) for obj in objects]
            state = jax.lax.switch(jax.random.randint(sk, (), 0, len(self.objects)), branches, state)
            return state
        elif self.num_objects == 1 and len(self.objects) == 1:
            state = super()._reset_internal(rng_key)
            if self.objects[0] == "can": state = remove_objects(state, ("milk", "bread", "cereal"))
            elif self.objects[0] == "milk": state = remove_objects(state, ("can", "bread", "cereal"))
            elif self.objects[0] == "bread": state = remove_objects(state, ("can", "milk", "cereal"))
            elif self.objects[0] == "cereal": state = remove_objects(state, ("can", "milk", "bread"))
            return state
        else:
            return super()._reset_internal(rng_key)
    
    def _over_bin(self, pos, bin_id):
        bin_low = self.model.body_pos[self.model.body("bin2").id][:2]
        bin_size = jnp.array((0.39, 0.49, 0.82)) / 2
        if bin_id == 0 or bin_id == 2:
            bin_low  = bin_low - jnp.array((bin_size[0], 0))
        if bin_id < 2:
            bin_low = bin_low - jnp.array((0, bin_size[0]))
        bin_high = bin_low + bin_size[:2]
        return jnp.logical_and(jnp.all(pos[:2] >= bin_low), jnp.all(pos[:2] <= bin_high))
        
    @jax.jit
    def reward(self, state, action, next_state):
        obs = self.observe(next_state, ManipulationTaskObs())
        objects_over_bins = jnp.array([self._over_bin(obj_pos, _TARGET_BIN_ID[obj]) \
                                       for obj, obj_pos in zip(self.objects, obs.object_pos)])

        object_z_pos = obs.object_pos[:,2]
        bin_z = self.model.body_pos[self.model.body("bin2").id][2]
        bin_z_check = jnp.logical_and(object_z_pos >= bin_z, object_z_pos <= bin_z + 0.1)

        objects_in_bins = jnp.logical_and(objects_over_bins, bin_z_check)
        return 0.5*jnp.mean(objects_over_bins) + 0.5*jnp.mean(objects_in_bins)

    def is_finished(self, state: SimulatorState) -> jax.Array:
        return super().is_finished(state)

    @jax.jit
    def render(self, state, config = None):
        return super().render(state, config)
    
    @jax.jit
    def observe(self, state, config : ObserveConfig | None = None):
        if config is None: config = ManipulationTaskObs()
        data = self.simulator.system_data(state)
        eef_id = self.model.site("gripper0_grip_site").id
        grip_site_id = self.model.site("gripper0_grip_site").id
        system_state = self.simulator.reduce_state(state)
        jacp, jacr = self.native_simulator.get_jacs(system_state, eef_id)

        robot = self._model_initializers[1][0]
        grip_qpos = self.reduce_state(state).qpos[robot.gripper_actuator_indices]
        if isinstance(config, ManipulationTaskObs):
            return ManipulationTaskObs(
                eef_pos=data.site_xpos[eef_id, :],
                eef_vel=jnp.dot(jacp, system_state.qvel),
                eef_ori_mat=data.site_xmat[eef_id, :].reshape([3, 3]),
                eef_ori_vel=jnp.dot(jacr, system_state.qvel),
                grip_qpos=grip_qpos,
                grip_site_pos=data.site_xpos[grip_site_id, :],
                object_pos=jnp.stack([
                    data.xpos[self.model.joint(_OBJECT_JOINT_MAP[obj]).id, :] for obj in self.objects
                ]),
                object_quat=jnp.stack([
                    data.xquat[self.model.joint(_OBJECT_JOINT_MAP[obj]).id, :] for obj in self.objects
                ])
            )
        elif isinstance(config, ManipulationTaskEEFPose):
            #return jnp.concatenate([data.site_xpos[eef_id, :], data.site_xmat[eef_id, :]])
            return data.site_xpos[eef_id, :].reshape(3,), data.site_xmat[eef_id, :].reshape([3, 3]), grip_qpos
        else:
            raise ValueError("Unsupported observation type")

_NUT_JOINT_MAP = {
    "round": "RoundNut_joint0",
    "square": "SquareNut_joint0"
}

_PEG_ID_NUT_MAP = {
    "square": 0,
    "round": 1,
}

@dataclass(kw_only=True)
class NutAssembly(RobosuiteEnv[SimulatorState]):
    num_objects: int = field(default=2, pytree_node=False)
    # the type of the nut
    objects: Sequence[str] = field(
        default=("round","square"),
        pytree_node=False
    )

    @property
    def env_name(self):
        return "NutAssembly"

    @property
    def _env_args(self):
        if self.num_objects == 1 and len(self.objects) == 1:
            return {"single_object_mode": 2, "nut_type": self.objects[0]}
        elif self.num_objects == 1:
            return {"single_object_mode": 1}
        elif self.num_objects == 2:
            return {}
        else:
            raise ValueError(f"Unsupported number of objects {self.num_objects}")
    
    def _reset_internal(self, rng_key : jax.Array) -> SystemState:
        # randomly select one of the nuts to remove
        def remove_nuts(state, names):
            qpos = state.qpos
            for n in names:
                qpos = _set_joint_qpos(self.model, _NUT_JOINT_MAP[n],
                    qpos, jnp.array([10, 10, 10, 1, 0, 0, 0])
                )
            return replace(state, qpos=qpos)

        if self.num_objects == 1 and len(self.objects) > 1:
            # randomly select a nut to remove
            objects = list(self.objects)
            rng_key, sk = jax.random.split(rng_key)
            state = super()._reset_internal(rng_key)
            branches = [partial(remove_nuts, names=[o for o in objects if o != obj]) for obj in objects]
            state = jax.lax.switch(jax.random.randint(sk, (), 0, len(objects)),
                branches, state)
        elif self.num_objects == 1 and len(self.objects) == 1:
            state = super()._reset_internal(rng_key)
            if self.objects[0] == "round": state = remove_nuts(state, ("square",))
            elif self.objects[0] == "square": state = remove_nuts(state, ("round",))
        else:
            state = super()._reset_internal(rng_key)
        return state

    @jax.jit
    def observe(self, state, config : ObserveConfig | None = None):
        if config is None: config = ManipulationTaskObs()
        if isinstance(config, ManipulationTaskObs):
            data = self.simulator.system_data(state)
            eef_id = self.model.site("gripper0_grip_site").id
            grip_site_id = self.model.site("gripper0_grip_site").id
            return ManipulationTaskObs(
                eef_pos=data.site_xpos[eef_id, :],
                eef_vel=data.cvel[eef_id, :3],
                eef_quat=data.xquat[eef_id, :],
                eef_rot_vel=data.cvel[eef_id, 3:],
                grip_site_pos=data.site_xpos[grip_site_id, :],
                object_pos=jnp.stack([
                    data.xpos[self.model.body(f"{obj.capitalize()}Nut_main").id, :] for obj in self.objects
                ]),
                object_quat=jnp.stack([
                    data.xquat[self.model.body(f"{obj.capitalize()}Nut_main").id, :] for obj in self.objects
                ])
            )
        else:
            raise ValueError("Unsupported observation type")

    @jax.jit
    def reward(self, state, action, next_state):
        peg_ids = [self.model.body("peg1").id, self.model.body("peg2").id]
        data = self.simulator.system_data(next_state)
        peg_pos = jnp.stack([data.xpos[peg_ids[_PEG_ID_NUT_MAP[obj]], :] for obj in self.objects])
        obs = self.observe(next_state, ManipulationTaskObs())
        over_peg = jnp.linalg.norm(peg_pos[:2] - obs.object_pos[:2], axis=-1) < 0.12
        z_pos = obs.object_pos[:,2]
        peg_z_range = jnp.logical_and(z_pos >= peg_pos[:,2] - 0.12, z_pos <= peg_pos[:,2] + 0.08)
        on_peg = jnp.logical_and(over_peg, peg_z_range)
        return 0.5*jnp.mean(over_peg) + 0.5*jnp.mean(on_peg)

@dataclass(kw_only=True)
class DoorOpen(RobosuiteEnv[SimulatorState]):
    @property
    def env_name(self):
        return "DoorOpen"

@dataclass
class ManipulationTaskEEFPose:
    pass

@dataclass
class ManipulationTaskObs:
    eef_pos: jax.Array = None # (n_robots, 3,) -- end-effector position
    eef_vel: jax.Array = None # (n_robots, 3,) -- end-effector velocity
    eef_ori_mat: jax.Array = None # (n_robots, 3, 3) -- end-effector orientation matrix
    eef_ori_vel: jax.Array = None # (n_robots, 3,) -- end-effector orientation velocity
    grip_qpos: jax.Array = None # (n_robots, 2,) -- gripper qpos

    grip_site_pos: jax.Array = None # (n_robots, 3,) -- gripper site position

    object_pos: jax.Array = None # (n_objects, 3) where n is the number of objects in the scene
    object_quat: jax.Array = None # (n_objects, 4) where n is the number of objects in the scene

@dataclass
class ManipulationTaskRelObs:
    eef_obj_rel_pos: jax.Array = None # (n_robots, n_objects, 3) -- relative position of the end-effector to the object
    obj_pos: jax.Array = None
    eef_ori_mat: jax.Array = None # (n_robots, 3, 3) -- end-effector orientation matrix
    object_quat: jax.Array = None # (n_objects, 4) -- object orientation
    grip_qpos: jax.Array = None # (n_robots, 2,) -- gripper qpos

@dataclass
class ManipulationTaskPosObs:
    eef_pos: jax.Array = None
    eef_ori_mat: jax.Array = None
    grip_qpos: jax.Array = None
    object_pos: jax.Array = None
    object_quat: jax.Array = None

@dataclass
class PositionalControlTransform(EnvTransform):
    k_p : jax.Array = jnp.array([1500]*6) # (6,) -- [0:3] corresponds to position, [3:6] corresponds to orientation
    k_d : jax.Array = jnp.array([240]*6) # (6,) -- [0:3] corresponds to position, [3:6] corresponds to orientation

    def apply(self, env):
        return PositionalControlEnv(env, self.k_p, self.k_d)
    
@dataclass
class PositionalObsTransform(EnvTransform):
    def apply(self, env):
        return PositionalObsEnv(env)
    
@dataclass
class RelPosObsTransform(EnvTransform):
    def apply(self, env):
        return RelPosObsEnv(env)
    

@dataclass
class PositionalControlEnv(EnvWrapper):
    k_p : jax.Array 
    k_d : jax.Array 

    def step(self, state, action, rng_key=None):
        obs = self.base.observe(state)
        if action is not None:
            # action_pos = jax.tree.map(lambda x: x[:,:3].reshape(3,), action)
            # action_ori_mat = jax.tree.map(lambda x: x[:,3:].reshape([3,3]), action)
            action_pos, action_ori_mat, grip_action = action
            action_pos = jnp.squeeze(action_pos)
            action_ori_mat = jnp.squeeze(action_ori_mat)
            data = self.simulator.system_data(state)
            robot = self._model_initializers[1][0]
            eef_id = self.model.site("gripper0_grip_site").id

            system_state = self.simulator.reduce_state(state)
            jacp, jacv = self.native_simulator.get_jacs(system_state, eef_id)
            J_pos = jnp.array(jacp.reshape((3, -1))[:, robot.qvel_indices])
            J_ori = jnp.array(jacv.reshape((3, -1))[:, robot.qvel_indices])
            
            mass_matrix = self.native_simulator.get_fullM(system_state)
            mass_matrix = jnp.reshape(mass_matrix, (len(data.qvel), len(data.qvel)))
            mass_matrix = mass_matrix[robot.qvel_indices, :][:, robot.qvel_indices]

            # Compute lambda matrices
            mass_matrix_inv = jnp.linalg.inv(mass_matrix)
            lambda_pos_inv = J_pos @ mass_matrix_inv @ J_pos.T
            lambda_ori_inv = J_ori @ mass_matrix_inv @ J_ori.T
            # take the inverses, but zero out small singular values for stability
            lambda_pos = jnp.linalg.pinv(lambda_pos_inv)
            lambda_ori = jnp.linalg.pinv(lambda_ori_inv)

            pos_error = action_pos - obs.eef_pos
            vel_pos_error = -obs.eef_vel

            eef_ori_mat = obs.eef_ori_mat
            ori_error = orientation_error(action_ori_mat, eef_ori_mat)
            vel_ori_error = -obs.eef_ori_vel

            F_r = self.k_p[:3] * pos_error + self.k_d[:3] * vel_pos_error
            Tau_r = self.k_p[3:] * ori_error + self.k_d[3:] * vel_ori_error
            compensation = data.qfrc_bias[robot.qvel_indices]

            #print(J_pos.T.shape, lambda_pos.shape, F_r.shape, J_ori.T.shape, lambda_ori.shape, Tau_r.shape, compensation.shape)
            torques = J_pos.T @ lambda_pos @ F_r + J_ori.T @ lambda_ori @ Tau_r + compensation
            a = jnp.zeros(self.model.nu, dtype=jnp.float32)
            a = a.at[robot.qvel_indices].set(torques)
            #jax.debug.print("J_pos: {J_pos}, J_ori: {J_ori}, lambda_pos: {lambda_pos}, lambda_ori: {lambda_ori}", J_pos=J_pos, J_ori=J_ori, lambda_pos=lambda_pos, lambda_ori=lambda_ori)
            #jax.debug.print("{s}, {t}, {u}", s=J_pos.T @ lambda_pos @ F_r, t=J_ori.T @ lambda_ori @ Tau_r, u=compensation)
            #jax.debug.print("{s}", s=a)

            a = a.at[jnp.squeeze(robot.gripper_actuator_indices)].set(jnp.squeeze(grip_action))

        else: 
            a = jnp.zeros(self.model.nu, dtype=jnp.float32)
        return self.base.step(state, a, None)


@dataclass
class PositionalObsEnv(EnvWrapper):
    def observe(self, state, config=None):
        if config is None: config = ManipulationTaskPosObs()
        if not isinstance(config, ManipulationTaskPosObs):
            return self.base.observe(state, config)
        obs = self.base.observe(state, ManipulationTaskObs())
        return ManipulationTaskPosObs(
            eef_pos=obs.eef_pos,
            eef_ori_mat=obs.eef_ori_mat,
            object_pos=obs.object_pos,
            object_quat=obs.object_quat
        )

@dataclass 
class RelPosObsEnv(EnvWrapper):
    def observe(self, state, config=None):
        if config is None: config = ManipulationTaskRelObs()
        if not isinstance(config, ManipulationTaskRelObs):
            return self.base.observe(state, config)
        obs = self.base.observe(state, ManipulationTaskObs())
        return ManipulationTaskRelObs(
            eef_obj_rel_pos=obs.object_pos - obs.eef_pos,
            obj_pos=obs.object_pos,
            eef_ori_mat=None, #obs.eef_ori_mat,
            object_quat=None, # obs.object_quat,
            grip_qpos=obs.grip_qpos
        )

environments = EnvironmentRegistry[RobosuiteEnv]()

# Pick and place environments
environments.register("pickplace", partial(PickAndPlace, 
    num_objects=4, objects=("can","milk", "bread", "cereal"), robots=("panda",)
))
environments.register("pickplace/random", partial(PickAndPlace, 
    num_objects=1, objects=("can","milk", "bread", "cereal"), robots=("panda",)
))
environments.register("pickplace/can", partial(PickAndPlace, 
    num_objects=1, objects=("can",), robots=("panda",)
))
environments.register("pickplace/milk", partial(PickAndPlace, 
    num_objects=1, objects=("milk",), robots=("panda",)
))
environments.register("pickplace/bread", partial(PickAndPlace, 
    num_objects=1, objects=("bread",), robots=("panda",)
))
environments.register("pickplace/cereal", partial(PickAndPlace, 
    num_objects=1, objects=("cereal",), robots=("panda",)
))

# Nut assembly environments
environments.register("nutassembly", partial(NutAssembly,
    robots=("panda",)
))
environments.register("nutassembly/random", partial(NutAssembly,
    num_objects=1, robots=("panda",)
))
environments.register("nutassembly/square", partial(NutAssembly,
    num_objects=1, objects=("square",), robots=("panda",)
))
environments.register("nutassembly/round", partial(NutAssembly,
    num_objects=1, objects=("round",), robots=("panda",)
))

# def _make_positional(**kwargs):
#     env = RobosuiteEnv(**kwargs)
#     return ChainedTransform([
#         PositionalControlTransform(),
#         PositionalObsTransform
#     ]).apply(env)
# environments.register("positional", _make_positional)

# Convert robosuite object/robot 
# initializers to jax-friendly format

@dataclass
class ObjectPlacement:
    pos: jax.Array
    quat: jax.Array
    joint_names: Sequence[str] = field(pytree_node=False)
    # the extents of the object (for exclusion purposes)
    object_horiz_radius : float = field(pytree_node=False)
    object_top_offset : float = field(pytree_node=False)
    object_bottom_offset : float = field(pytree_node=False)

@dataclass
class RobotInitializer:
    init_qpos: jax.Array
    joint_indices: np.ndarray = field(pytree_node=False)
    qpos_indices: np.ndarray = field(pytree_node=False)
    qvel_indices: np.ndarray = field(pytree_node=False)
    gripper_actuator_indices: np.ndarray = field(pytree_node=False)
    noiser: Callable[[jax.Array, jax.Array], jax.Array] | None = None

    def reset(self, rng_key : jax.Array | None, state: SystemState) -> SystemState:
        robot_qpos = self.init_qpos
        if rng_key is not None and self.noiser is not None:
            robot_qpos = self.noiser(rng_key, robot_qpos)
        qpos = state.qpos.at[self.qpos_indices].set(robot_qpos)
        return replace(state, qpos=qpos)

    @staticmethod
    def from_robosuite(robot: Any, randomized : bool = True) -> "RobotInitializer":
        noiser = None
        if robot.initialization_noise["type"] == "gaussian":
            def noiser(rng_key, qpos):
                m = robot.initialization_noise["magnitude"]
                return qpos + m * jax.random.normal(rng_key, qpos.shape)
        elif robot.initialization_noise["type"] == "uniform":
            def noiser(rng_key, qpos):
                m = robot.initialization_noise["magnitude"]
                return qpos + m * jax.random.uniform(rng_key, qpos.shape, minval=-1, maxval=1)
        return RobotInitializer(
            robot.init_qpos,
            np.array(robot._ref_joint_indexes),
            np.array(robot._ref_joint_pos_indexes),
            np.array(robot._ref_joint_vel_indexes),
            np.array(robot._ref_joint_gripper_actuator_indexes),
            noiser if randomized else None
        )

@dataclass
class ObjectInitializer:
    # each sampler is a function from a random key to a dictionary of object placements
    object_samplers: Sequence[Callable[[jax.Array], dict[str, ObjectPlacement]]]

    def generate(self, rng_key : jax.Array) -> dict[str, ObjectPlacement]:
        keys = jax.random.split(rng_key, len(self.object_samplers))
        placement = {}
        for r, v in zip(keys, self.object_samplers):
            placement.update(v(r, placement))
        return placement

    # for a given state, update the object placements

    @staticmethod
    def update_state(model : mujoco.MjModel, state : SystemState, 
                     placements : Iterable[ObjectPlacement]) -> SystemState:
        qpos = state.qpos
        for v in placements:
            for j in v.joint_names:
                qpos = _set_joint_qpos(model, j, qpos, jnp.concatenate((v.pos, v.quat)))
        return replace(state, qpos=qpos)

    @staticmethod
    def from_robosuite(sampler : Any, sample_args : dict | None = None) -> "ObjectInitializer":
        from robosuite.utils.placement_samplers import (
            SequentialCompositeSampler,
            UniformRandomSampler
        )
        samplers = []
        if isinstance(sampler, SequentialCompositeSampler):
            for s in sampler.samplers.values():
                initializer = ObjectInitializer.from_robosuite(s, sampler.sample_args)
                samplers.extend(initializer.object_samplers)
        else:
            object_info = list(
                (o.name, [o.joints[0]], o.horizontal_radius, o.bottom_offset, o.top_offset) for o in sampler.mujoco_objects 
                if "visual" not in o.name.lower()
            ) # we only want the physical objects, not the visual (which are part of the model)
            if isinstance(sampler, UniformRandomSampler):
                min_pos, max_pos = (
                    jnp.array([sampler.x_range[0], sampler.y_range[0]]),
                    jnp.array([sampler.x_range[1], sampler.y_range[1]])
                )
                if sampler.rotation is None:
                    rot_range = (0, 2 * jnp.pi)
                elif isinstance(sampler.rotation, collections.abc.Iterable):
                    rot_range = (min(sampler.rotation), max(sampler.rotation))
                else:
                    rot_range = None
                    rotation = sampler.rotation
                rotation_axis = sampler.rotation_axis
                z_offset = jnp.array(sampler.z_offset)

                # offset by the reference position
                reference_pos = jnp.array(sampler.reference_pos)
                min_pos = min_pos + reference_pos[:2]
                max_pos = max_pos + reference_pos[:2]
                z_offset = z_offset + reference_pos[2]
                on_top = sample_args.get("on_top", True) if sample_args else True

                def uniform_sampler(rng_key: jax.Array, fixtures: dict[str, ObjectPlacement]) -> dict[str, ObjectPlacement]:
                    placements = {}
                    keys = jax.random.split(rng_key, len(object_info))
                    for rk, (name, joint_names, horiz_radius, bot_off, top_off) in zip(keys, object_info):
                        z_pos = z_offset[None]
                        if on_top: z_pos = z_pos - (bot_off[-1])[None]

                        def sampler(sk):
                            return jax.random.uniform(sk, (2,), 
                                minval=min_pos + horiz_radius, maxval=max_pos - horiz_radius
                            )
                        xy_k, a_k = jax.random.split(rk)
                        xy_pos = _loop_sample(placements, horiz_radius, top_off, bot_off, z_pos, xy_k, sampler)
                        pos = jnp.concatenate((xy_pos, z_pos))
                        rot = jax.random.uniform(a_k, (), minval=rot_range[0], maxval=rot_range[1]) \
                            if rot_range is not None else rotation
                        quat = _angle_to_quat(rot, rotation_axis)
                        placement = ObjectPlacement(
                            pos=pos,
                            quat=quat,
                            joint_names=joint_names,
                            object_horiz_radius=horiz_radius,
                            object_bottom_offset=bot_off,
                            object_top_offset=top_off
                        )
                        placements[name] = placement
                    return placements
                samplers.append(uniform_sampler)
            else:
                raise ValueError(f"Unsupported sampler type {sampler}")
        return ObjectInitializer(samplers)

def _loop_sample(placements, horiz_radius, 
            top_offset, bot_offset, z, rng_key, sample_fn):
    if not placements:
        return sample_fn(rng_key)
    # stack the positions of the objects that have been placed already
    o_xy_pos = jnp.stack(list(p.pos[:2] for p in placements.values()))
    o_z_pos = jnp.array(list(p.pos[2] for p in placements.values()))
    o_horiz_radius = jnp.array(list(p.object_horiz_radius for p in placements.values()))
    o_top_offset = jnp.array(list(p.object_top_offset[-1] for p in placements.values()))

    def sample_pos(loop_state):
        i, r, _ = loop_state
        sk, r = jax.random.split(r)
        return i + 1, r, sample_fn(sk)
    def is_invalid(loop_state):
        i, _, xy_pos = loop_state
        too_close = jnp.sum(jnp.square(xy_pos - o_xy_pos), axis=-1) <= (horiz_radius + o_horiz_radius) ** 2
        underneath = o_z_pos - z <= o_top_offset - bot_offset[-1]
        return jnp.logical_and(i < 64, jnp.any(jnp.logical_and(too_close, underneath)))
    loop_state = sample_pos((
        jnp.zeros((), jnp.uint32), 
        rng_key, jnp.zeros((2,), jnp.float32)
    ))
    _, _, xy_pos = jax.lax.while_loop(is_invalid, sample_pos, loop_state)
    return xy_pos

# random utilities
def _set_joint_qpos(model, joint_name, qpos, joint_val):
    id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    addr = model.jnt_qposadr[id]
    type = model.jnt_type[id]

    if type == mujoco.mjtJoint.mjJNT_FREE:
        return qpos.at[addr:addr+7].set(joint_val)
    elif type == mujoco.mjtJoint.mjJNT_BALL:
        return qpos.at[addr:addr+4].set(joint_val)
    elif type == mujoco.mjtJoint.mjJNT_SLIDE:
        return qpos.at[addr:addr+1].set(joint_val)
    elif type == mujoco.mjtJoint.mjJNT_HINGE:
        return qpos.at[addr:addr+1].set(joint_val)
    else:
        raise ValueError(f"Unsupported joint type {type}")

def _angle_to_quat(rot_angle, rotation_axis):
    if rotation_axis == "x":
        return jnp.array([jnp.cos(rot_angle / 2), jnp.sin(rot_angle / 2), 0, 0])
    elif rotation_axis == "y":
        return jnp.array([jnp.cos(rot_angle / 2), 0, jnp.sin(rot_angle / 2), 0])
    elif rotation_axis == "z":
        return jnp.array([jnp.cos(rot_angle / 2), 0, 0, jnp.sin(rot_angle / 2)])
    else:
        raise ValueError(f"Unsupported rotation axis {rotation_axis}")

def _setup_macros():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import robosuite
    import os
    import shutil
    base_path = robosuite.__path__[0]
    macros_path = os.path.join(base_path, "macros.py")
    macros_private_path = os.path.join(base_path, "macros_private.py")
    if os.path.exists(macros_private_path):
        return
    shutil.copyfile(macros_path, macros_private_path)