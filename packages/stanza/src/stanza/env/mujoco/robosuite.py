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
from stanza.env.mujoco import (
    MujocoEnvironment,
    SystemState, SimulatorState
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
    def model_initializers(self) -> tuple[mujoco.MjModel, Sequence[RobotInitializer], ObjectInitializer]:
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
        env.reset()
        model =  env.sim.model._model
        robots = tuple(RobotInitializer.from_robosuite(r) for r in env.robots)
        initializer = ObjectInitializer.from_robosuite(env.placement_initializer)
        return model, robots, initializer
    
    @property
    def model(self):
        return self.model_initializers[0]

    @jax.jit
    def reset(self, rng_key : jax.Array) -> SimulatorState:
        state = SystemState(
            time=jnp.zeros((), jnp.float32),
            qpos=self.simulator.qpos0,
            qvel=self.simulator.qvel0,
            act=self.simulator.act0
        )
        # get the model, robot initializers, and object initializers
        model, robots, initializer = self.model_initializers
        keys = jax.random.split(rng_key, len(robots) + 1)
        r_keys, i_key = keys[:-1], keys[-1]
        for rk, r in zip(r_keys, robots):
            state = r.reset(rk, state)
        placements = initializer.generate(i_key)
        state = ObjectInitializer.update_state(model, state, placements.values())

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


    # For pick and place, use camera 1 by default
    def render(self, state, config = None):
        return super().render(state, config)
    
    @jax.jit
    def observe(self, state, config : ObserveConfig = None):
        if config is None: config = PickPlaceObs()
        if isinstance(config, PickPlaceObs):
            data = self.simulator.data(state)
            eef_id = self.model.body("gripper0_eef").id
            return PickPlaceObs(
                eef_pos=data.xpos[eef_id, :],
                eef_vel=data.cvel[eef_id, :3],
                eef_quat=data.xquat[eef_id, :],
                eef_rot_vel=data.cvel[eef_id, 3:],
                object_pos=jnp.stack([
                    data.xpos[self.model.body(f"{obj.capitalize()}_main").id, :] for obj in self.objects
                ]),
                object_quat=jnp.stack([
                    data.xquat[self.model.body(f"{obj.capitalize()}_main").id, :] for obj in self.objects
                ])
            )
        else:
            raise ValueError("Unsupported observation type")
    
    def get_action(self, state):
        #return self.observe(state).eef_pos
        return jnp.zeros(self.model.nu)
        
@dataclass
class PickPlaceObs:
    eef_pos: jax.Array = None # (3,) -- end-effector position
    eef_vel: jax.Array = None # (3,) -- end-effector velocity
    eef_quat: jax.Array = None # (4,) -- end-effector quaternion
    eef_rot_vel: jax.Array = None # (3,) -- end-effector angular velocity

    object_pos: jax.Array = None # (n, 3) where n is the number of objects in the scene
    object_quat: jax.Array = None # (n, 4) where n is the number of objects in the scene

@dataclass
class PickPlacePosObs:
    eef_pos: jax.Array = None
    eef_quat: jax.Array = None
    object_pos: jax.Array = None
    object_quat: jax.Array = None

@dataclass
class PickPlaceEulerObs:
    pass

@dataclass
class PositionalControlTransform(EnvTransform):
    #TODO
    def apply(self, env):
        return PositionalControlEnv(env)
    
@dataclass
class PositionalObsTransform(EnvTransform):
    def apply(self, env):
        return PositionalObsEnv(env)

@dataclass
class PositionalControlEnv(EnvWrapper):
    #TODO
    def step(self, state, action, rng_key=None):
        a = jnp.zeros(self.base.model.nu)
        return self.base.step(state, a, None)

@dataclass
class PositionalObsEnv(EnvWrapper):
    def observe(self, state, config=None):
        if config is None: config = PickPlacePosObs()
        if not isinstance(config, PickPlacePosObs):
            return self.base.observe(state, config)
        obs = self.base.observe(state, PickPlaceObs())
        return PickPlacePosObs(
            eef_pos=obs.eef_pos,
            eef_quat=obs.eef_quat,
            object_pos=obs.object_pos,
            object_quat=obs.object_quat
        )
    
    def get_action(self, state):
        #return self.observe(state).eef_pos
        return jnp.zeros(self.model.nu)

@dataclass(kw_only=True)
class NutAssembly(RobosuiteEnv[SimulatorState]):
    @property
    def env_name(self):
        return "NutAssembly"

@dataclass(kw_only=True)
class DoorOpen(RobosuiteEnv[SimulatorState]):
    @property
    def env_name(self):
        return "DoorOpen"

environments = EnvironmentRegistry[RobosuiteEnv]()
environments.register("pickplace", partial(PickAndPlace, 
    num_objects=4, objects=("can","milk", "bread", "cereal"), robots=("panda",)
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

environments.register("nutassembly", partial(NutAssembly,
    robots=("panda",)
))

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

    def reset(self, rng_key : jax.Array | None, state: SystemState) -> SystemState:
        robot_qpos = self.init_qpos
        if rng_key is not None:
            pass
        qpos = state.qpos.at[self.joint_indices].set(robot_qpos)
        return replace(state, qpos=qpos)

    @staticmethod
    def from_robosuite(robot: Any) -> "RobotInitializer":
        return RobotInitializer(
            robot.init_qpos,
            np.array(robot._ref_joint_pos_indexes)
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
            for (s, sa) in zip(sampler.samplers.values(), sampler.sample_args):
                initializer = ObjectInitializer.from_robosuite(s, sa)
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

                def uniform_sampler(rng_key: jax.Array, fixtures: dict[str, ObjectPlacement]) -> dict[str, ObjectPlacement]:
                    placements = {}
                    keys = jax.random.split(rng_key, len(object_info))
                    for rk, (name, joint_names, horz_radius, bot_off, top_off) in zip(keys, object_info):
                        xy_k, a_k = jax.random.split(rk)
                        xy_pos = jax.random.uniform(xy_k, (2,), 
                            minval=min_pos, maxval=max_pos
                        )
                        z_pos = z_offset[None]
                        pos = jnp.concatenate((xy_pos, z_pos))
                        rot = jax.random.uniform(a_k, (), minval=rot_range[0], maxval=rot_range[1]) \
                            if rot_range is not None else rotation
                        quat = _angle_to_quat(rot, rotation_axis)
                        placement = ObjectPlacement(
                            pos=pos,
                            quat=quat,
                            joint_names=joint_names,
                            object_horiz_radius=horz_radius,
                            object_bottom_offset=bot_off,
                            object_top_offset=top_off
                        )
                        placements[name] = placement
                    return placements
                samplers.append(uniform_sampler)
            else:
                raise ValueError(f"Unsupported sampler type {sampler}")
        return ObjectInitializer(samplers)

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