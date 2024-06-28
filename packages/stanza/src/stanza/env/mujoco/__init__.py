from .core import (
    SystemState, SystemData, 
    Simulator, MujocoEnvironment,
)

from .. import EnvironmentRegistry, from_module

environments = EnvironmentRegistry[MujocoEnvironment]()
# env_registry.defer(register_module(".pusht", "env_registry"))
environments.extend("pusht", from_module(".pusht", "environments"))