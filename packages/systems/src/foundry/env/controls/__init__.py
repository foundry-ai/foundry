from foundry.env import EnvironmentRegistry, Environment
from foundry.util.registry import from_module

environments = EnvironmentRegistry[Environment]()
environments.extend("linear", from_module(".linear", "environments"))
environments.extend("pendulum", from_module(".pendulum", "environments"))
environments.extend("quadrotor2d", from_module(".quadrotor2d", "environments"))
