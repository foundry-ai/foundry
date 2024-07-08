import jax
import jax.numpy as jnp

from stanza.dataclasses import dataclass, field
from stanza.util import jax_static_property
from stanza.env import (
    EnvWrapper, RenderConfig, 
    ImageRender, SequenceRender,
    HtmlRender, Environment,
    EnvironmentRegistry
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

    @jax_static_property
    def xml(self):
        _setup_macros()
        import robosuite as suite
        env = suite.make(
            env_name=self.env_name,
            robots=[ROBOT_NAME_MAP[r] for r in self.robots],
            has_renderer=False,
            use_camera_obs=False,
        )
        return env.model.get_xml()
    
    @jax.jit
    def render(self, config: RenderConfig, state: SimulatorState) -> jax.Array:
        config = config or ImageRender(width=256, height=256)
        if isinstance(config, ImageRender):
            pass
        else: return super().render(config, state)

# Pick and Place environment:
OBJEcT_NAME_MAP = {
    "can": "Can", "milk": "Milk", 
    "bread": "Bread", "cereal": "Cereal"
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

    def reset(self, rng_key) -> SimulatorState:
        return self.full_state(SystemState(
            time=jnp.zeros((), jnp.float32),
            qpos=self.simulator.qpos0,
            qvel=self.simulator.qvel0,
            act=self.simulator.act0
        ))

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