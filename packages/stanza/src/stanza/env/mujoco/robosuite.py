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

    # For pick and place, use camera 1 by default
    def render(self, state, config = None):
        return super().render(state, config)

_NUT_JOINT_MAP = {
    "round": "RoundNut_joint0",
    "square": "SquareNut_joint0"
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

@dataclass(kw_only=True)
class DoorOpen(RobosuiteEnv[SimulatorState]):
    @property
    def env_name(self):
        return "DoorOpen"

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
    noiser: Callable[[jax.Array, jax.Array], jax.Array] | None = None

    def reset(self, rng_key : jax.Array | None, state: SystemState) -> SystemState:
        robot_qpos = self.init_qpos
        if rng_key is not None and self.noiser is not None:
            robot_qpos = self.noiser(rng_key, robot_qpos)
        qpos = state.qpos.at[self.joint_indices].set(robot_qpos)
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
            np.array(robot._ref_joint_pos_indexes),
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