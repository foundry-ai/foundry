from stanza.env import EnvironmentRegistry, Environment
from stanza.util.registry import from_module

environments = EnvironmentRegistry[Environment]()
environments.extend(from_module(".linear", "environments"))
environments.extend(from_module(".pendulum", "environments"))
environments.extend(from_module(".quadrotor2d", "environments"))
