import importlib
import inspect

import jax
import jax.tree_util
import jax.numpy as jnp

from stanza.envs import Environment
from typing import Any

__ENV_BUILDERS = {}

def create(env_type, **kwargs):
    env_path = env_type.split("/")
    # register buildres if empty
    builder = __ENV_BUILDERS[env_path[0]]()
    return builder(env_type, **kwargs)

# Register them lazily so we don't
# import dependencies we don't actually use
# i.e the appropriate submodule will be imported
# for the first time during create()
def register_lazy(name, module_name):
    frm = inspect.stack()[1]
    pkg = inspect.getmodule(frm[0]).__name__
    def make_env_constructor():
        mod = importlib.import_module(module_name, package=pkg)
        return mod.builder
    __ENV_BUILDERS[name] = make_env_constructor