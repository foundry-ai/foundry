from .core import (
    State, Action, Observation, Render,
    Environment, EnvironmentRegistry,
    RenderConfig, ImageRender,
    HtmlRender, 
    ObserveConfig,
    EnvWrapper,
)

from foundry.util.registry import from_module

environments = EnvironmentRegistry[Environment]()
# env_registry.defer(register_module(".pusht", "env_registry"))
environments.extend("controls", from_module(".controls", "environments"))
environments.extend("mujoco", from_module(".mujoco", "environments"))

def create(path: str, /, **kwargs):
    return environments.create(path, **kwargs)