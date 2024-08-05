import jax
import jax.numpy as jnp

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
from typing import Sequence

# handles the XML generation, rendering

ROBOT_NAME_MAP = {"panda": "Panda"}

@dataclass(kw_only=True)
class RobosuiteEnv(MujocoEnvironment[SimulatorState]):
    robots: Sequence[str] = field(pytree_node=False)

    @property
    def _env_args(self):
        return {}

    # override the model to use the robosuite-initialized model
    # which may differ slightly from the one generated from the XML
    # (e.g. visualization options)
    @jax_static_property
    def model(self):
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
        return env.sim.model._model

    # use the "frontview" camera for rendering
    # if not specified
    def render(self, state, config = None):
        config = config or ImageRender(width=256, height=256)
        # custom image rendering for robosuite
        # which disables the collision geometry visualization
        if isinstance(config, ImageRender):
            data = self.simulator.data(state)
            camera = config.camera if config.camera is not None else "frontview"
            # render only the visual geometries
            # do not include the collision geometries
            return self.native_simulator.render(
                data, config.width, config.height, (False, True), camera
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

    def reset(self, rng_key : jax.Array) -> SimulatorState:
        return self.full_state(SystemState(
            time=jnp.zeros((), jnp.float32),
            qpos=self.simulator.qpos0,
            qvel=self.simulator.qvel0,
            act=self.simulator.act0
        ))

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


environments = EnvironmentRegistry[RobosuiteEnv]()
environments.register("can", partial(PickAndPlace, 
    num_objects=1, objects=("can",), robots=("panda",)
))

# random utilities

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