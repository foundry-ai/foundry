from stanza.env import EnvironmentRegistry, Environment
from stanza.util.registry import from_module

environments = EnvironmentRegistry[Environment]()
environments.extend("pusht", from_module(".pusht", "environments"))