from stanza.env import EnvironmentRegistry, Environment
from stanza.util.registry import from_module

from mujoco import mjx

@dataclass
class SystemState:
    q: jax.Array
    qd: jax.Array

@dataclass
class MjxState:
    data: mjx.Data

environments = EnvironmentRegistry[Environment]()
environments.extend("pusht", from_module(".pusht", "environments"))